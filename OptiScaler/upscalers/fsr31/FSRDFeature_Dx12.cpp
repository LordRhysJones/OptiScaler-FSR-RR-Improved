#include "pch.h"
#include <nvsdk_ngx_defs_dlssd.h>
#include <DirectXMath.h>
#include "NVNGX_Parameter.h"
#include "hooks/Streamline_Hooks.h"
#include "FSRDFeature_Dx12.h"
#include "shaders/fsrd_preprocess/FSRDPreprocessor_Dx12.h"
#include "MathUtils.h"

using namespace DirectX;
using namespace OptiMath;

using FSRDConvDesc = FSRDPreprocessor_Dx12::ConversionDesc;
using FSRDCompDesc = FSRDPreprocessor_Dx12::CompositionDesc;

/**
 * @brief Retrieves a matrix from the given parameter table. Matrices used by DLSS are in column-major
 * order, but DirectXMath operations assume row-major. Appropriate for passing to DirectX shaders, but not for
 * CPU-side operations without transposing.
 */
static bool TryGetNGXMatrixTranspose(const NVSDK_NGX_Parameter& ngxParams, const char* key, DirectX::XMMATRIX& outValue)
{
    float* pMat = nullptr;

    if (ngxParams.Get(key, (void**) &pMat) == NVSDK_NGX_Result_Success && pMat != nullptr)
    {
        memcpy_s(&outValue, sizeof(DirectX::XMMATRIX), pMat, sizeof(float) * 16);
        return true;
    }
    else
        return false;
}

/**
 * @brief Retrieves a matrix from the given parameter table and transposes it for CPU-side
 * operations with DirectXMath.
 */
static bool TryGetNGXMatrix(const NVSDK_NGX_Parameter& ngxParams, const char* key, DirectX::XMMATRIX& outValue)
{
    if (TryGetNGXMatrixTranspose(ngxParams, key, outValue))
    {
        outValue = XMMatrixTranspose(outValue);
        return true;
    }
    else
        return false;
}

template <typename T>
static bool TryGetLoggedResource(const NVSDK_NGX_Parameter& ngxParams, const char* key, T*& outValue)
{
    const bool success = TryGetNGXVoidPointer(ngxParams, key, outValue);

    if (success)
        LOG_DEBUG("{} exists..", key);
    else
        LOG_ERROR("{} is missing!!", key);

    return success;
}

/**
 * @brief Calculates vertical FOV according to: FOVv = 2 * arctan( 1 / M22 )
 * @param proj View to Clip / Perspective projection matrix
 * @return Vertical field of view in radians
 */
static float GetVertFovFromProjectionMatrixRad(const XMMATRIX& proj)
{
    return float(2.0 * (std::atan(1.0 / (double) proj.r[1].m128_f32[1])));
}

/**
 * @brief Calculates horizontal FOV according to: FOVh = 2 * arctan( 1 / M11 )
 * @param proj View to Clip / Perspective projection matrix
 * @return Horizontal field of view in radians
 */
static float GetHorzFovFromProjectionMatrixRad(const XMMATRIX& proj)
{
    return float(2.0 * (std::atan(1.0 / (double) proj.r[0].m128_f32[0])));
}

/**
 * @brief Calculates aspect ratio (width / height) as AR = M22 / M11
 * @param proj View to Clip / Perspective projection matrix
 * @return Aspect ratio as an fp32 decimal e.g. 1.778
 */
static float GetAspectRatioFromProjectionMatrix(const XMMATRIX& proj)
{
    return proj.r[1].m128_f32[1] / proj.r[0].m128_f32[0];
}

static XMFLOAT3 GetFloat3(const XMVECTOR& vec4)
{
    XMFLOAT3 vec3 = {};
    XMStoreFloat3(&vec3, vec4);
    return vec3;
}

static XMVECTOR GetColumn(const XMMATRIX& mat, int col)
{
    return { mat.r[0].m128_f32[col], mat.r[1].m128_f32[col], mat.r[2].m128_f32[col], 0 };
}

static void SetColumn(const XMVECTOR& vec, int col, XMMATRIX& mat)
{ 
    mat.r[0].m128_f32[col] = vec.m128_f32[0]; 
    mat.r[1].m128_f32[col] = vec.m128_f32[1]; 
    mat.r[2].m128_f32[col] = vec.m128_f32[2]; 
    mat.r[3].m128_f32[col] = vec.m128_f32[3]; 
}

static XMFLOAT3 GetFloat3Column(const XMMATRIX& mat, int col)
{
    return { mat.r[0].m128_f32[col], mat.r[1].m128_f32[col], mat.r[2].m128_f32[col] };
}

static FfxApiFloatCoords3D GetFloat3ColumnFFX(const XMMATRIX& mat, int col)
{
    return { mat.r[0].m128_f32[col], mat.r[1].m128_f32[col], mat.r[2].m128_f32[col] };
}

static FfxApiFloatCoords3D GetFloat3FFX(const XMVECTOR& vec4)
{
    FfxApiFloatCoords3D vec3 = {};
    XMStoreFloat3(reinterpret_cast<XMFLOAT3*>(&vec3), vec4);
    return vec3;
}

static const FfxApiFloatCoords3D& GetFloat3FFX(const XMFLOAT3& vec3)
{
    return *reinterpret_cast<const FfxApiFloatCoords3D*>(&vec3);
}

static ID3D12Resource* GetD3D12ResFromFFX(const FfxApiResource& resource)
{
    return static_cast<ID3D12Resource*>(resource.resource);
}

struct ViewPlanes
{
    float nearPlane;
    float farPlane;
    bool isInfinite;
    bool isRightHanded;
};

static ViewPlanes GetViewPlanes(const DirectX::XMMATRIX& projection, bool isInverted)
{
    ViewPlanes planes;
    // View to clip
    float A = projection.r[2].m128_f32[2];
    float B = projection.r[2].m128_f32[3];
    float W = projection.r[3].m128_f32[2];

    float infiniteCheckVal = isInverted ? A : (A - W);
    planes.isInfinite = std::abs(infiniteCheckVal) < 1e-6f;

    // W is the term that produces clip w from view z, so it is -1 for right handed projections and +1
    // for left handed ones. It survives reverse-Z and infinite-far variants untouched, which the A and B
    // terms do not.
    planes.isRightHanded = W < 0.0f;

    if (isInverted)
    {
        // Inverted: Near is at D=1, Far is at D=0
        // 1 = A/W + B/(n*W) -> n = B / (W - A)
        planes.nearPlane = std::abs(B / (W - A));

        // 0 = A/W + B/(f*W) -> f = -B / A
        planes.farPlane = std::abs(-B / A);
    }
    else
    {
        // Standard: Near is at D=0, Far is at D=1
        // 0 = A/W + B/(n*W) -> n = -B / A
        planes.nearPlane = std::abs(-B / A);

        // 1 = A/W + B/(f*W) -> f = B / (W - A)
        planes.farPlane = std::abs(B / (W - A));
    }

    return planes;
}

using FSRDConvFlags = FSRDPreprocessor_Dx12::ConvFlags;
using FSRDCompFlags = FSRDPreprocessor_Dx12::CompFlags;

enum class DebugModes : uint64_t
{
    None = 0,
    DenoiserBypass = 1,
    UpscalerBypass = 2,
    RawColor = 3,
    DlssBias = 4,
    DlssColorBeforeParticles = 5,
    DlssColorBeforeTransparency = 6,
    DlssTransparencyLayer = 7,
    FfxDebug = 8,

    ConversionDebug = FSRDConvFlags::Debug,
    ConversionDebugMask = FSRDConvFlags::DebugModeMask,

    OutRadiance = FSRDConvFlags::DebugOutRadiance,

    InSpecHitDist = FSRDConvFlags::DebugInSpecHitDist,
    InMotion = FSRDConvFlags::DebugInMotion,
    InNormals = FSRDConvFlags::DebugInNormals,
    InRoughness = FSRDConvFlags::DebugInRoughness,
    InDiffAlbedo = FSRDConvFlags::DebugInDiffAlbedo,
    InSpecAlbedo = FSRDConvFlags::DebugInSpecAlbedo,

    OutLinearDepth = FSRDConvFlags::DebugOutLinearDepth,
    OutMotion = FSRDConvFlags::DebugOutMotion,
    OutNormals = FSRDConvFlags::DebugOutNormals,
    OutSpecAlbedo = FSRDConvFlags::DebugOutSpecAlbedo,
    OutDiffAlbedo = FSRDConvFlags::DebugOutDiffAlbedo,

    OutDepthDelta = FSRDConvFlags::DebugOutDepthDelta,
    NormDepth = FSRDConvFlags::DebugNormDepth,
    AlbedoError = FSRDConvFlags::DebugAlbedoError,

    FloorVariance = FSRDConvFlags::DebugFloorVariance,
    FloorColor = FSRDConvFlags::DebugFloorColor,

    CompositionDebugOffset = 16u,
    CompositionDebug = (uint64_t) FSRDCompFlags::Debug << CompositionDebugOffset,
    CompositionDebugMask = (uint64_t)FSRDCompFlags::DebugModeMask,

    Correlation = (uint64_t)FSRDCompFlags::DebugCorrelation << CompositionDebugOffset,
    SkipSignal = (uint64_t) FSRDCompFlags::DebugSkipSignal << CompositionDebugOffset,
    DenoiserOutput = (uint64_t) FSRDCompFlags::DebugDenoiserOutput << CompositionDebugOffset,
    Signal1 = (uint64_t) FSRDCompFlags::DebugSignal1 << CompositionDebugOffset,
    Signal2 = (uint64_t) FSRDCompFlags::DebugSignal2 << CompositionDebugOffset,
};

static FSRDConvFlags GetConvDebugFlags(DebugModes mode) 
{ 
    uint32_t flags = uint32_t(mode);
    flags &= uint32_t(DebugModes::ConversionDebugMask);
    return FSRDConvFlags(flags);
}

static FSRDCompFlags GetCompDebugFlags(DebugModes mode) 
{ 
    uint64_t flags = uint64_t(mode);
    flags >>= uint64_t(DebugModes::CompositionDebugOffset);
    flags &= uint64_t(DebugModes::CompositionDebugMask);
    return FSRDCompFlags(flags);
}

using ModeNamePair = std::pair<const char*, uint64_t>;
constexpr auto kDebugModes = std::to_array<ModeNamePair>(
{
    { "None", (uint64_t) DebugModes::None },
    { "DebugOverview", (uint64_t) DebugModes::FfxDebug },

    { "DenoiserBypass", (uint64_t) DebugModes::DenoiserBypass },
    { "UpscalerBypass", (uint64_t) DebugModes::UpscalerBypass },
    { "DenoiserOutput", (uint64_t) DebugModes::DenoiserOutput },
    { "SkipSignal", (uint64_t) DebugModes::SkipSignal },

    { "RawColor", (uint64_t) DebugModes::RawColor },
    { "DlssBias", (uint64_t) DebugModes::DlssBias },
    { "DlssColorBeforeParticles", (uint64_t) DebugModes::DlssColorBeforeParticles },
    { "DlssColorBeforeTransparency", (uint64_t) DebugModes::DlssColorBeforeTransparency },
    { "DlssTransparencyLayer", (uint64_t) DebugModes::DlssTransparencyLayer },

    { "InMotionVectors", (uint64_t) DebugModes::InMotion },
    { "InNormals", (uint64_t) DebugModes::InNormals },
    { "InRoughness", (uint64_t) DebugModes::InRoughness },
    { "InSpecHitDist", (uint64_t) DebugModes::InSpecHitDist },
    { "InDiffAlbedo", (uint64_t) DebugModes::InDiffAlbedo },
    { "InSpecAlbedo", (uint64_t) DebugModes::InSpecAlbedo },

    { "OutRadiance", (uint64_t) DebugModes::OutRadiance },
    { "OutLinearDepth", (uint64_t) DebugModes::OutLinearDepth },
    { "OutMotionVectors", (uint64_t) DebugModes::OutMotion },
    { "OutNormals", (uint64_t) DebugModes::OutNormals },
    { "OutSpecAlbedo", (uint64_t) DebugModes::OutSpecAlbedo },
    { "OutDiffAlbedo", (uint64_t) DebugModes::OutDiffAlbedo },
    { "OutDepthDelta", (uint64_t) DebugModes::OutDepthDelta },
    { "NormDepth", (uint64_t) DebugModes::NormDepth },

    { "AlbedoError", (uint64_t) DebugModes::AlbedoError },
    { "Correlation", (uint64_t) DebugModes::Correlation },

    { "FloorVariance", (uint64_t) DebugModes::FloorVariance },
    { "FloorColor", (uint64_t) DebugModes::FloorColor },
    
    { "Signal1", (uint64_t) DebugModes::Signal1 },
    { "Signal2", (uint64_t) DebugModes::Signal2 },
});


bool FSRDFeatureDx12::s_isHWDepth = false;
bool FSRDFeatureDx12::s_isRoughnessPacked = false;

FSRDFeatureDx12::FSRDFeatureDx12(uint32_t InHandleId, NVSDK_NGX_Parameter* InParameters) :
    FSR31FeatureDx12(InHandleId, InParameters),
    IFeature(InHandleId, SetParameters(InParameters)),
    _pDenoiserCtx(nullptr),
    _denoiserCtxDesc({}),
    _denoiserSettings({}),
    _deltaTime(0.0f),
    _cameraFovVertical(0.0f),
    _debugDepthBounds({}),
    _convDesc({}),
    _isRightHanded(false)
{
    _moduleLoaded = FfxApiProxy::IsDenoiserReady();

    if (_moduleLoaded)
        LOG_INFO("amd_fidelityfx_denoiser_dx12.dll methods loaded!");
    else
        LOG_ERROR("can't load amd_fidelityfx_denoiser_dx12.dll methods!");
}

FSRDFeatureDx12::~FSRDFeatureDx12() 
{
    if (State::Instance().isShuttingDown)
        return;

    DestroyDenoiserContext();
}

bool FSRDFeatureDx12::InitFSR3(const NVSDK_NGX_Parameter* InParameters)
{
    LOG_FUNC();

    // Init upscaler first - borrow some init boilerplate and some cfg
    if (FSR31FeatureDx12::InitFSR3(InParameters))
    {
        SetInit(false);

        LOG_DEBUG("FSR Ray Regeneration Initializing");
        _name = OptiTexts::FSR_RR_Name;

        if (int value; InParameters->Get(NVSDK_NGX_Parameter_Use_HW_Depth, &value) == NVSDK_NGX_Result_Success)
            s_isHWDepth = value == NVSDK_NGX_DLSS_Depth_Type_HW;

        if (int value; InParameters->Get(NVSDK_NGX_Parameter_DLSS_Roughness_Mode, &value) == NVSDK_NGX_Result_Success)
            s_isRoughnessPacked = value == NVSDK_NGX_DLSS_Roughness_Mode_Packed;

        LOG_INFO("DLSSD Flags HWDepth: {} - IsRoughnessPacked: {}", s_isHWDepth, s_isRoughnessPacked);

        if (!CreateDenoiserContext())
            return false;

        LOG_INFO("FSR Ray Regeneration Initialized");

        SetInit(true);
        return true;
    }
 
    return false;
}

bool FSRDFeatureDx12::CreateDenoiserContext() 
{
    ScopedSkipSpoofing skipSpoofing {};
    auto& state = State::Instance();
    const auto& cfg = *Config::Instance();

    if (!QueryDenoiserVersions())
        return false;

    state.ffxDenoiserUpscalerVersion = Version();
    parse_version(state.ffxDenoiserVersionNames[cfg.FfxDenoiserIndex.value_or_default()]);

    ffxOverrideVersion vidOverride =
    {
        .header = { .type = FFX_API_DESC_TYPE_OVERRIDE_VERSION },
        .versionId = state.ffxDenoiserVersionIds[cfg.FfxDenoiserIndex.value_or_default()]
    };
    // Create context
    // Backend desc
    ffxCreateBackendDX12Desc backendDesc = 
    { 
        .header = 
        { 
            .type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12,
            .pNext = &vidOverride.header // Chain override into backend desc
        },
        .device = Device
    };    
    // Chain: ContextDesc -> BackendDesc -> OverrideVersion
    // Per-lobe indirect specular and diffuse radiance, without a dominant light source
    _denoiserCtxDesc =
    {
        .header = 
        { 
            .type = FFX_API_CREATE_CONTEXT_DESC_TYPE_DENOISER,
            // Chain backend desc into context desc
            .pNext = &backendDesc.header
        },
        .version = FFX_DENOISER_VERSION,
        .maxRenderSize = { RenderWidth(), RenderHeight() },
        // DLSS-RR hands us separate specular and diffuse radiance, which map onto the two indirect
        // signals. 1.2 dropped the fused single-signal mode that Mode 1 used to rely on.
        .signalFlags = FFX_DENOISER_SIGNAL_INDIRECT_SPECULAR | FFX_DENOISER_SIGNAL_INDIRECT_DIFFUSE,
        // DLSS-RR inputs are fully traced every frame, never checkerboarded.
        .checkerboardSignalFlags = FFX_DENOISER_SIGNAL_NONE,
        .flags = 0
    };

#ifdef _DEBUG
    LOG_INFO("Debug checking enabled for denoiser!");
    _denoiserCtxDesc.flags |= FFX_DENOISER_ENABLE_DEBUGGING | FFX_DENOISER_ENABLE_VALIDATION;
#endif

    // Create the denoiser context
    {   
        ScopedSkipHeapCapture skipHeapCapture {};
        auto ret = FfxApiProxy::D3D12_CreateContext(&_pDenoiserCtx, &_denoiserCtxDesc.header, NULL);

        if (ret != FFX_API_RETURN_OK)
        {
            LOG_ERROR("_denoiserCtx error: {0}", FfxApiProxy::ReturnCodeToString(ret));
            return false;
        }
    }

    // Query default settings
    SetDefaultConfiguration();

    // Create DLSS-RR to FSR-RR input converter
    FSRDConvShader = std::make_unique<FSRDPreprocessor_Dx12>("FSRD Converter", Device);

    if (!FSRDConvShader->IsInit())
        return false;

    if (!FSRDConvShader->SetMaxRenderSize(_denoiserCtxDesc.maxRenderSize.width, _denoiserCtxDesc.maxRenderSize.height))
        return false;

    return true;
}

bool FSRDFeatureDx12::QueryDenoiserVersions() 
{
    ScopedSkipSpoofing skipSpoofing {};
    auto& state = State::Instance();

    // Get version count
    uint64_t versionCount = 0;
    ffxQueryDescGetVersions queryVersionsDesc = 
    { 
        .header = { .type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS },
        .createDescType = FFX_API_EFFECT_ID_DENOISER,
        .device = Device,
        .outputCount = &versionCount
    };
    FfxApiProxy::D3D12_Query(nullptr, &queryVersionsDesc.header);

    state.ffxDenoiserVersionIds.resize(versionCount);
    state.ffxDenoiserVersionNames.resize(versionCount);

    state.ffxDenoiserDebugModes.clear();
    state.ffxDenoiserDebugModeNames.clear();

    for (const auto& mode : kDebugModes)
    {
        state.ffxDenoiserDebugModes.push_back(mode.second);
        state.ffxDenoiserDebugModeNames.emplace(mode.second, mode.first);
    }

    if (versionCount == 0)
    {
        LOG_ERROR("No FSR-RR denoisers were found.");
        return false;
    }
    else
        LOG_DEBUG("Found {} versions of FSR-RR", versionCount);

    LOG_DEBUG("Initialising FSR denoiser context");

    // Get version IDs
    queryVersionsDesc.versionIds = state.ffxDenoiserVersionIds.data();
    queryVersionsDesc.versionNames = state.ffxDenoiserVersionNames.data();
    FfxApiProxy::D3D12_Query(nullptr, &queryVersionsDesc.header);

    return true;
}

void FSRDFeatureDx12::DestroyDenoiserContext() 
{
    if (_pDenoiserCtx != nullptr)
        FfxApiProxy::D3D12_DestroyContext(&_pDenoiserCtx, nullptr);
}

void FSRDFeatureDx12::UpdateSize() 
{
    // FSR-RR doesn't currently have proper DRS support. The example implementation 
    // reinits on resolution change as well.
    const bool needsReInit = 
        _denoiserCtxDesc.maxRenderSize.width != RenderWidth() ||
        _denoiserCtxDesc.maxRenderSize.height != RenderHeight();

    if (needsReInit)
    {
        LOG_INFO(
            "Reinitializing FSR-RR for resolution change. "
            "Previous: {} x {}, New: {} x {}",
            _denoiserCtxDesc.maxRenderSize.width, _denoiserCtxDesc.maxRenderSize.height,
            RenderWidth(), RenderHeight());

        DestroyDenoiserContext();
        CreateDenoiserContext();
    }
}

bool FSRDFeatureDx12::Evaluate(ID3D12GraphicsCommandList* InCommandList, NVSDK_NGX_Parameter* InParameters) 
{
    LOG_FUNC();

    if (!IsInited())
        return false;

    auto& state = State::Instance();
    auto& cfg = *Config::Instance();
    const auto& inParams = *InParameters;

    UpdateSize();

    const auto dbgMode = static_cast<DebugModes>(cfg.FfxDenoiserDebugMode.value_or_default());
    const bool isDebugVis = (uint32_t)dbgMode & (uint32_t) DebugModes::ConversionDebug;
    const bool isDebugComp = ((uint64_t)dbgMode & (uint64_t)DebugModes::CompositionDebug);
    const bool isFfxDebug = dbgMode == DebugModes::FfxDebug;
    const bool hasAnyDebug = (dbgMode != DebugModes::None);

    // Denoise is bypassed if we are debugging something OTHER than the final outputs
    const bool isDenoiseBypassed = !isFfxDebug && !isDebugComp && 
        hasAnyDebug && dbgMode != DebugModes::DenoiserOutput && dbgMode != DebugModes::UpscalerBypass;

    // Upscale is bypassed if we are in a debug mode that isn't the DenoiserBypass (final raw)
    const bool isUpscaleBypassed = hasAnyDebug && dbgMode != DebugModes::DenoiserBypass;

    // Validate helper features
    if (!RCAS->IsInit())
        cfg.RcasEnabled.set_volatile_value(false);
    if (!OutputScaler->IsInit())
        cfg.OutputScalingEnabled.set_volatile_value(false);

    _isInReset = false;

    if (uint32_t value = 0; inParams.Get(NVSDK_NGX_Parameter_Reset, &value) == NVSDK_NGX_Result_Success)
        _isInReset = value > 0;

    // Denoiser start
    //
    // 1.2 takes one desc per denoised lobe instead of a single fused signal desc. DLSS-RR gives us
    // separate specular and diffuse radiance, so both indirect lobes are dispatched.
    ffxDispatchDescDenoiserIndirectSpecular specularDesc = {};
    ffxDispatchDescDenoiserIndirectDiffuse diffuseDesc = {};
    ffxDispatchDescDenoiser denoiserDesc = {};
    bool isDenoiserReady = false;

    // Pull configuration and input buffers for DLSS-RR from the param table, convert and
    // repack input buffers into intermediate FSR-RR input buffers, and configure descriptors.
    if (!PrepareDenoiserInput(InCommandList, *InParameters, denoiserDesc, specularDesc, diffuseDesc))
        return false;

    // Dispatch denoiser
    if (!isDenoiseBypassed)
    {
        ffxDispatchDescDenoiserDebugView dispatchDebugView = {};

        if (isFfxDebug)
        {
            ID3D12Resource* dstTex;
            TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_Output, dstTex);

            dispatchDebugView =
            {
                .header = { .type = FFX_API_DISPATCH_DESC_TYPE_DENOISER_DEBUG_VIEW },
                .output = ffxApiGetResourceDX12(dstTex, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS),
                .outputSize = { TargetWidth(), TargetHeight() },
                .mode = FFX_API_DENOISER_DEBUG_VIEW_MODE_OVERVIEW,
                .viewportIndex = 0
            };

            // Tail of the chain built by PrepareDenoiserInput
            diffuseDesc.header.pNext = &dispatchDebugView.header;
        }

        isDenoiserReady = DispatchDenoiser(InCommandList, denoiserDesc);

        if (!isDenoiserReady)
            return false;

        // Compose denoised signals
        FSRDCompDesc compDesc = 
        { 
            .DstTexSize = _convDesc.RenderSize,
            .CorrelationBias = cfg.FfxDenoiserCorrelationBias.value_or_default(),
            .Flags = (uint32_t)GetCompDebugFlags(dbgMode)
        };

        TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_Color, compDesc.InRawColor);
        TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_DLSSD_ColorBeforeParticles, compDesc.InColorBeforeParticles);

        if (!isFfxDebug && !FSRDConvShader->DispatchComposition(InCommandList, compDesc))
            return false;

        isDenoiserReady = true;
    }

    // Upscaler start
    if (!isUpscaleBypassed)
    {
        ffxDispatchDescUpscale upscalerDesc = {};

        if (!PrepareUpscalerInput(InCommandList, inParams, upscalerDesc))
            return false;

        // Override upscaler config
        if (isDenoiserReady)
        {
            upscalerDesc.color = ffxApiGetResourceDX12(FSRDConvShader->GetCompositionOutput());
            upscalerDesc.cameraFovAngleVertical = _cameraFovVertical;
            upscalerDesc.frameTimeDelta = _deltaTime;
        }

        // Sets optional, configurable resource barriers
        FSR31FeatureDx12::SetConfigurableBarriers(InCommandList);

        bool isUpscalerReady = DispatchUpscaler(InCommandList, upscalerDesc);

        // Post-Process
        if (isUpscalerReady)
            PostProcess(InCommandList, inParams);

        // Cleanup
        FSR31FeatureDx12::ResetConfigurableBarriers(InCommandList);
    }
    else if (!isFfxDebug) // Debug visualization
    {
        ID3D12Resource* srcTex = nullptr;

        if (dbgMode == DebugModes::DlssColorBeforeParticles)
            TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_DLSSD_ColorBeforeParticles, srcTex);
        else if (dbgMode == DebugModes::DlssColorBeforeTransparency)
            TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_DLSSD_ColorBeforeTransparency, srcTex);
        else if (dbgMode == DebugModes::DlssTransparencyLayer)
            TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_DLSS_TransparencyLayer, srcTex);
        else if (dbgMode == DebugModes::DlssBias)
            TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, srcTex);
        else if (dbgMode == DebugModes::RawColor)
            TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_Color, srcTex);
        else if (isDebugVis)
            srcTex = GetD3D12ResFromFFX(specularDesc.signal.input);
        else
            srcTex = FSRDConvShader->GetCompositionOutput();

        ID3D12Resource* dstTex;

        if (!srcTex || !TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_Output, dstTex))
        {
            _frameCount++;
            return true;
        }

        FSRDConvShader->Blit(InCommandList, srcTex, dstTex);
    }

    _frameCount++;
    return isDenoiserReady || isDenoiseBypassed;
}

bool FSRDFeatureDx12::PrepareDenoiserInput(ID3D12GraphicsCommandList* InCommandList, const NVSDK_NGX_Parameter& inParams,
    ffxDispatchDescDenoiser& dispatchDesc, ffxDispatchDescDenoiserIndirectSpecular& specularDesc,
    ffxDispatchDescDenoiserIndirectDiffuse& diffuseDesc)
{
    const auto& cfg = *Config::Instance(); 
    const auto& slData = State::Instance().slLastConstants;

    // Gather DLSS-RR input buffers for conversion and repacking for FSR-RR
    if (!PrepareDenoiseConvInput(inParams))
        return false;   

    if (!ConvertDenoiserBuffers(InCommandList))
        return false;

    // Camera position, taken from viewMatrix^-1. The matrices are stored in the column-vector convention
    // (see TryGetNGXMatrix), so translation lives in the fourth column.
    const XMFLOAT3 camPos = GetFloat3Column(_invViewMatrix, 3);

    // Camera transforms
    //
    // 1.2 takes the view and projection matrices directly rather than a decomposed camera basis plus
    // fov/aspect/near/far. It wants row-major storage with row-vector multiplication - DirectXMath's
    // native form - so transpose out of the column-vector convention these are stored in.
    //
    // The conversion shader emits unsigned linear depth and an unsigned depth delta, whereas 1.2 documents
    // both as signed and consistent with the view matrix it is handed. Rather than re-signing those buffers
    // (which would mean rebuilding the committed shader blobs), make view space left handed so that
    // positive-forward *is* the correct sign, and pair it with a left handed projection below.
    XMMATRIX viewFFX = XMMatrixTranspose(_viewMatrix);

    // Negates the third column, flipping view space z to positive-forward
    if (_isRightHanded)
        viewFFX = XMMatrixMultiply(viewFFX, XMMatrixScaling(1.0f, 1.0f, -1.0f));

    // Rebuild the projection canonically instead of forwarding the title's own. Games hand us reverse-Z,
    // infinite-far and right handed variants, but the denoiser only needs the frustum they describe - and
    // NearPlane/FarPlane have already been sanitised for stability in ConvertDenoiserBuffers. Reusing them
    // keeps the projection, linearDepthBounds and the shader's validity cutoff all in agreement.
    _cameraFovVertical = GetVertFovFromProjectionMatrixRad(_projMatrix);

    const XMMATRIX projFFX = XMMatrixPerspectiveFovLH(_cameraFovVertical,
                                                      GetAspectRatioFromProjectionMatrix(_projMatrix),
                                                      _convDesc.NearPlane, _convDesc.FarPlane);

    // Pack dispatch configuration
    dispatchDesc =
    {
        .commandList = InCommandList,
        .motionVectorScale = { 1.0f, 1.0f, 1.0f },
        // Camera movement since last frame (PreviousPosition - CurrentPosition)
        .cameraPositionDelta = { (_lastCamPos.x - camPos.x), (_lastCamPos.y - camPos.y), (_lastCamPos.z - camPos.z) },
        // Bounds on *absolute* linear depth, so these stay positive regardless of handedness. Signal
        // texels outside the range are passed through untouched rather than denoised.
        .linearDepthBounds = { .min = _convDesc.NearPlane, .max = _convDesc.FarPlane },
        .renderSize = { RenderWidth(), RenderHeight() },
        .frameIndex = (uint32_t)_frameCount,
        .flags = FFX_DENOISER_DISPATCH_NON_GAMMA_ALBEDO
    };

    // FfxApiMatrix4x4 is 16 contiguous row-major floats, matching XMFLOAT4X4
    static_assert(sizeof(FfxApiMatrix4x4) == sizeof(XMFLOAT4X4));
    XMStoreFloat4x4(reinterpret_cast<XMFLOAT4X4*>(&dispatchDesc.view), viewFFX);
    XMStoreFloat4x4(reinterpret_cast<XMFLOAT4X4*>(&dispatchDesc.projection), projFFX);

    // Populate resources and build the desc chain: dispatch -> indirectSpecular -> indirectDiffuse
    FSRDConvShader->GetSignals(dispatchDesc, specularDesc, diffuseDesc);

    if (_isInReset)
        dispatchDesc.flags |= FFX_DENOISER_DISPATCH_RESET;

    // Update camera position for next frame
    _lastCamPos = camPos;

    // 1.2 dropped deltaTime from the dispatch desc, but the upscaler handoff in Evaluate() still needs it
    if (!TryGetToggleableNGXParam(inParams, OptiKeys::FSR_FrameTimeDelta, cfg.FsrUseFsrInputValues, _deltaTime))
    {
        if (inParams.Get(NVSDK_NGX_Parameter_FrameTimeDeltaInMsec, &_deltaTime) !=
                NVSDK_NGX_Result_Success || _deltaTime < 1.0f)
        {
            _deltaTime = (float)GetDeltaTime();
        }
    }

    // Motion Vector Scaling
    // Scaling must result in UV space vectors, unlike FSR/DLSS pixel space vectors
    float MVScaleX = 1.0f, MVScaleY = 1.0f;

    if (inParams.Get(NVSDK_NGX_Parameter_MV_Scale_X, &MVScaleX) == NVSDK_NGX_Result_Success &&
        inParams.Get(NVSDK_NGX_Parameter_MV_Scale_Y, &MVScaleY) == NVSDK_NGX_Result_Success)
    {
        dispatchDesc.motionVectorScale.x = MVScaleX / dispatchDesc.renderSize.width;
        dispatchDesc.motionVectorScale.y = MVScaleY / dispatchDesc.renderSize.height;
    }

    float jitterX = 0.0f, jitterY = 0.0f;
    inParams.Get(NVSDK_NGX_Parameter_Jitter_Offset_X, &jitterX);
    inParams.Get(NVSDK_NGX_Parameter_Jitter_Offset_Y, &jitterY);

    // Convert from pixel to NDC jitter. Inline AMD docs incorrectly claim this is "expressed in screen pixels".
    // The RR 1.0 and 1.1 reference implementations use NDC jitter. Fucking clowns.
    dispatchDesc.jitterOffsets.x = 2.0f * (jitterX / (float) RenderWidth());
    dispatchDesc.jitterOffsets.y = -2.0f * (jitterY / (float) RenderHeight());

    LOG_DEBUG("Jitter NDC [{:.6f}, {:.6f}]", dispatchDesc.jitterOffsets.x, dispatchDesc.jitterOffsets.y);

    return true;
}

bool FSRDFeatureDx12::PrepareDenoiseConvInput(const NVSDK_NGX_Parameter& inParams)
{
    const auto& slData = State::Instance().slLastConstants;

    // Gather DLSS-RR input buffers for conversion and repacking for FSR-RR
    bool isReady = true;

    // Standard TSR buffers
    if (!TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_Color, _convDesc.Resources.InColor))
        isReady = false;
    if (!TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_MotionVectors, _convDesc.Resources.InMotionVectors))
        isReady = false;
    if (!TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_Depth, _convDesc.Resources.InDepth) && LowResMV())
        isReady = false;

    // DLSSD-specific buffers
    if (!TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_GBuffer_Normals, _convDesc.Resources.InNormals))
        isReady = false;

    // If roughness is not packed into normals, then this texture is mandatory.
    // This value should be available in one of these two buffers in any DLSS-RR implementation.
    if (!TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_GBuffer_Roughness, _convDesc.Resources.InRoughness) &&
        !s_isRoughnessPacked)
    {
        LOG_WARN("Expected unpacked roughness buffer from DLSS-RR. Defaulting to packed roughness...");
        s_isRoughnessPacked = true;
    }

    if (!TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_DiffuseAlbedo, _convDesc.Resources.InDiffAlbedo))
        isReady = false;

    if (!TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_SpecularAlbedo, _convDesc.Resources.InSpecAlbedo))
        isReady = false;

    // Optional resources - these two are legitimately allowed to be absent (many DLSS-RR game
    // integrations don't wire them up, or only supply them some frames). _convDesc is a persistent
    // member reused every frame, and TryGetNGXVoidPointer/TryGetLoggedResource only overwrite the
    // output pointer on a *successful* Get() - they leave it untouched on failure. Without an
    // explicit reset here, a frame where the game doesn't supply one of these left the pointer
    // holding whatever resource it referenced last time it succeeded (possibly several frames ago,
    // an entire loading screen ago, or never - in a different game that never provides it at all).
    // That stale/dangling resource then got bound and sampled as if it were this frame's real data,
    // producing incoherent per-pixel garbage that changes underneath it as the game reuses that GPU
    // memory for something else - which reads as noise that never settles, concentrated on whatever
    // surfaces actually use that input (specular hit distance only affects smooth/reflective
    // surfaces, which is exactly where the reported sparkly artifacting near light sources showed
    // up). Explicitly clearing to nullptr first makes a failed fetch this frame correctly bind a
    // proper null SRV (CreateSRV already handles nullptr safely) instead of silently reusing old data.
    _convDesc.Resources.InBiasMask = nullptr;
    TryGetNGXVoidPointer(inParams, NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, _convDesc.Resources.InBiasMask);

    // Optional. Specular hit distance can be used with mode-2 denoising to track movement inside reflections, 
    // in addition to primary motion tracking for the surface and camera.
    _convDesc.Resources.InSpecHitDist = nullptr;
    TryGetLoggedResource(inParams, NVSDK_NGX_Parameter_DLSSD_SpecularHitDistance, _convDesc.Resources.InSpecHitDist);
    
    // Get DLSSD matrices and derive related values
    // World to view/camera space (V)
    _prevViewMatrix = _viewMatrix;
    _viewMatrix = {};

    if (!TryGetNGXMatrix(inParams, NVSDK_NGX_Parameter_DLSS_WORLD_TO_VIEW_MATRIX, _viewMatrix))
    {
        if (StreamlineHooks::isSetConstantsHooked())
        {
            SetColumn(XMLoadFloat3((XMFLOAT3*) &slData.cameraRight), 0, _invViewMatrix);
            SetColumn(XMLoadFloat3((XMFLOAT3*) &slData.cameraUp), 1, _invViewMatrix);
            SetColumn(XMLoadFloat3((XMFLOAT3*) &slData.cameraFwd), 2, _invViewMatrix);
            SetColumn(XMLoadFloat3((XMFLOAT3*) &slData.cameraPos), 3, _invViewMatrix);
            _invViewMatrix.r[3].m128_f32[3] = 1.0f;

            _viewMatrix = XMMatrixInverse(nullptr, _invViewMatrix);
        }
        else
        {
            LOG_ERROR("View matrix missing! Denoiser not ready.");
            isReady = false;
        }
    }
    else
    {
        // Camera rotation and position
        _invViewMatrix = XMMatrixInverse(nullptr, _viewMatrix);
    }

    // Perspective projection matrix (P)
    _projMatrix = {};

    if (!TryGetNGXMatrix(inParams, NVSDK_NGX_Parameter_DLSS_VIEW_TO_CLIP_MATRIX, _projMatrix))
    {
        if (StreamlineHooks::isSetConstantsHooked())
        {
            if (slData.cameraFOV != sl::INVALID_FLOAT && slData.cameraNear != slData.cameraFar)
            {
                // These measurements are supposed to be in radians, but some titles supply degrees.
                // Valid FOV in radians never exceeds PI. Realistic FOV in degrees is basically never in the single
                // digits.
                const float fov = (slData.cameraFOV < 4.0f) ? slData.cameraFOV : GetRadiansFromDeg(slData.cameraFOV);
                const float nearPlane = slData.cameraNear;
                const float farPlane = slData.cameraFar;
                _isRightHanded = slData.cameraViewToClip[2].w < 0.0f;

                // Actual SL view to clip matrix isn't reliable. This is harder to fuck up.
                if (_isRightHanded)
                    _projMatrix = XMMatrixPerspectiveFovRH(fov, slData.cameraAspectRatio, nearPlane, farPlane);
                else
                    _projMatrix = XMMatrixPerspectiveFovLH(fov, slData.cameraAspectRatio, nearPlane, farPlane);

                _projMatrix = XMMatrixTranspose(_projMatrix);
            }
        }
        else
        {
            LOG_ERROR("Projection matrix missing! Denoiser not ready.");
            isReady = false;
        }
    }

    return isReady;
}

bool FSRDFeatureDx12::ConvertDenoiserBuffers(ID3D12GraphicsCommandList* InCommandList)
{
    const uint32_t dbgMode = (uint32_t)Config::Instance()->FfxDenoiserDebugMode.value_or_default(); 
    const auto& cfg = *Config::Instance(); 
    const auto& slData = State::Instance().slLastConstants;

    // Prepare input converter
    _convDesc.RenderSize = 
    { 
        (float) RenderWidth(), (float) RenderHeight(), 
        1.0f / (float) RenderWidth(), 1.0f / (float) RenderHeight()
    };
    _convDesc.Flags = (uint32_t) FSRDConvFlags::NonGammaAlbedo | (dbgMode & (uint32_t) FSRDConvFlags::DebugModeMask);
    _convDesc.FloorIsolation = cfg.FfxDenoiserFloorIsolation.value_or_default();

    if (s_isRoughnessPacked)
        _convDesc.Flags |= (uint32_t) FSRDConvFlags::IsRoughnessPacked;

    // Store in column major order for GPU
    XMStoreFloat4x4(&_convDesc.InvViewMatrix, XMMatrixTranspose(_invViewMatrix));

    // Inverse perspective projection
    const XMMATRIX invProjMatrix = XMMatrixInverse(nullptr, _projMatrix);
    XMStoreFloat4x4(&_convDesc.InvProjMatrix, XMMatrixTranspose(invProjMatrix));

    // Previous world to view for linear depth delta
    XMStoreFloat4x4(&_convDesc.PrevViewMatrix, XMMatrixTranspose(_prevViewMatrix));

    // Near and far planes
    //
    // Infinite-far-plane (reverse-Z) projection matrices make A (m22) tend to 0, which is exactly what
    // GetViewPlanes() uses to detect the infinite case. Deriving FarPlane as abs(-B/A) in that situation
    // divides by a near-zero value, producing a huge value that is numerically unstable and can jitter
    // slightly frame-to-frame due to floating point noise in the source matrix. Since FarPlane directly
    // sets the cutoff used to decide whether a pixel is "valid" or "skipped" in the conversion shader,
    // that instability caused distant geometry (walls, roofs, anything near the far end of depth range)
    // to flicker between the two paths - producing noise and broken temporal accumulation (ghosting) on
    // far-away surfaces specifically. Fall back to a stable, configurable far plane whenever the matrix
    // can't give us a trustworthy one.
    const ViewPlanes planes = GetViewPlanes(_projMatrix, DepthInverted());
    _convDesc.NearPlane = planes.nearPlane;

    // Only the Streamline fallback in PrepareDenoiseConvInput knows its own handedness up front; when the
    // title supplies real matrices this is the only place it gets determined.
    _isRightHanded = planes.isRightHanded;

    const bool farPlaneIsUnstable =
        planes.isInfinite || !std::isfinite(planes.farPlane) || planes.farPlane <= planes.nearPlane;

    _convDesc.FarPlane = farPlaneIsUnstable ? Config::Instance()->FsrCameraFar.value_or_default() : planes.farPlane;

    if (!s_isHWDepth)
        _convDesc.Flags |= (uint32_t) FSRDConvFlags::IsDepthLinear;

    LOG_DEBUG("Distpaching FSRD Input Converter");

    // Dispatch resource converter. Outputs are automatically transitioned for reading.
    if (!FSRDConvShader->DispatchConversion(InCommandList, _convDesc))
        return false;

    return true;
}

static bool TryUpdateOption(const CustomOptional<float>& cfgValue, float& currentValue)
{
    if (cfgValue.value_or_default() != currentValue)
    {
        currentValue = cfgValue.value_or_default();
        return true;
    }
    else
        return false;
}

bool FSRDFeatureDx12::DispatchDenoiser(ID3D12GraphicsCommandList* InCommandList,
                                       const ffxDispatchDescDenoiser& dispatchDesc)
{
    auto& state = State::Instance();
    const auto& cfg = *Config::Instance();
    bool cfgChanged = false;

    if (TryUpdateOption(cfg.FfxDenoiserDisocThreshold, _denoiserSettings.m_DisocclusionThreshold))
        ApplyConfiguration(FFX_API_CONFIGURE_DENOISER_KEY_DISOCCLUSION_THRESHOLD);
    if (TryUpdateOption(cfg.FfxDenoiserCrossBlNormStr, _denoiserSettings.m_CrossBilateralNormalStrength))
        ApplyConfiguration(FFX_API_CONFIGURE_DENOISER_KEY_CROSS_BILATERAL_NORMAL_STRENGTH);
    if (TryUpdateOption(cfg.FfxDenoiserStabilityBias, _denoiserSettings.m_StabilityBias))
        ApplyConfiguration(FFX_API_CONFIGURE_DENOISER_KEY_STABILITY_BIAS);
    if (TryUpdateOption(cfg.FfxDenoiserMaxRadiance, _denoiserSettings.m_MaxRadiance))
        ApplyConfiguration(FFX_API_CONFIGURE_DENOISER_KEY_MAX_RADIANCE);
    if (TryUpdateOption(cfg.FfxDenoiserRadianceClip, _denoiserSettings.m_RadianceClipStdK))
        ApplyConfiguration(FFX_API_CONFIGURE_DENOISER_KEY_RADIANCE_CLIP_STD_K);
    if (TryUpdateOption(cfg.FfxDenoiserGaussKernRelax, _denoiserSettings.m_GaussianKernelRelaxation))
        ApplyConfiguration(FFX_API_CONFIGURE_DENOISER_KEY_GAUSSIAN_KERNEL_RELAXATION);

    // Keep the depth debug view normalized against the same range the dispatch is given
    if (_debugDepthBounds.x != _convDesc.NearPlane || _debugDepthBounds.y != _convDesc.FarPlane)
    {
        _debugDepthBounds = { _convDesc.NearPlane, _convDesc.FarPlane };
        ApplyDebugViewDepthBounds(_debugDepthBounds.x, _debugDepthBounds.y);
    }

    LOG_DEBUG("Dispatching FSR-RR...");
    const ffxReturnCode_t result = FfxApiProxy::D3D12_Dispatch(&_pDenoiserCtx, &dispatchDesc.header);

    if (result != FFX_API_RETURN_OK)
    {
        LOG_ERROR("_dispatch error: {0}", FfxApiProxy::ReturnCodeToString(result));

        if (result == FFX_API_RETURN_ERROR_RUNTIME_ERROR)
        {
            LOG_WARN("Trying to recover by recreating the feature");
            state.changeBackend[Handle()->Id] = true;
        }

        return false;
    }

    return true;
}

void FSRDFeatureDx12::SetDefaultConfiguration()
{
    for (int i = 0; i < DenoiserConfiguration::kCount; i++)
        SetDefaultConfiguration(DenoiserConfiguration::GetIndexKey(i));
}

ffxReturnCode_t FSRDFeatureDx12::SetDefaultConfiguration(FfxApiConfigureDenoiserKey key)
{
    ffxQueryDescDenoiserGetDefaultKeyValue queryDesc = 
    {
        .header = { .type = FFX_API_QUERY_DESC_TYPE_DENOISER_GET_DEFAULT_KEYVALUE }, 
        .key = (uint64_t)key, 
        .count = 1u,
        .data = &_denoiserSettings.GetMember(key)
    };

    const ffxReturnCode_t code = FfxApiProxy::D3D12_Query(&_pDenoiserCtx, &queryDesc.header);
    return code;
}

ffxReturnCode_t FSRDFeatureDx12::ApplyConfiguration(FfxApiConfigureDenoiserKey key)
{
    ffxConfigureDescDenoiserKeyValue configureDesc =
    {
        .header = { .type = FFX_API_CONFIGURE_DESC_TYPE_DENOISER_KEYVALUE },
        .key = (uint64_t)key,
        .count = 1u,
        .data = &_denoiserSettings.GetMember(key)
    };

    const ffxReturnCode_t code = FfxApiProxy::D3D12_Configure(&_pDenoiserCtx, &configureDesc.header);
    return code;
}

ffxReturnCode_t FSRDFeatureDx12::ApplyDebugViewDepthBounds(float nearPlane, float farPlane)
{
    // Key 7 is an FfxApiFloatBounds rather than a float, so it can't live in the float-keyed
    // _denoiserSettings array. It only normalizes the absolute linear depth debug view.
    const FfxApiFloatBounds bounds = { .min = nearPlane, .max = farPlane };

    ffxConfigureDescDenoiserKeyValue configureDesc =
    {
        .header = { .type = FFX_API_CONFIGURE_DESC_TYPE_DENOISER_KEYVALUE },
        .key = (uint64_t) FFX_API_CONFIGURE_DENOISER_KEY_DEBUG_VIEW_LINEAR_DEPTH_BOUNDS,
        .count = 1u,
        .data = &bounds
    };

    return FfxApiProxy::D3D12_Configure(&_pDenoiserCtx, &configureDesc.header);
}