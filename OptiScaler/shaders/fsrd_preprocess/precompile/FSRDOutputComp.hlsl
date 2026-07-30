#include "FSRDPreprocessCommon.hlsli"

#define MainRS \
    "RootFlags(0), " \
    "CBV(b0), " \
    "DescriptorTable(SRV(t0, numDescriptors = 7), visibility = SHADER_VISIBILITY_ALL), " \
    "DescriptorTable(UAV(u0, numDescriptors = 1), visibility = SHADER_VISIBILITY_ALL), " \
    "StaticSampler(s0, " \
        "filter = FILTER_MIN_MAG_MIP_LINEAR, " \
        "addressU = TEXTURE_ADDRESS_CLAMP, " \
        "addressV = TEXTURE_ADDRESS_CLAMP, " \
        "addressW = TEXTURE_ADDRESS_CLAMP, " \
        "visibility = SHADER_VISIBILITY_ALL)"

// Dispatch config
#define THREAD_GROUP_SIZE_X     8
#define THREAD_GROUP_SIZE_Y     8
#define NUM_THREADS             (THREAD_GROUP_SIZE_X * THREAD_GROUP_SIZE_Y)

static const uint2 s_ThreadGroupSize = uint2(THREAD_GROUP_SIZE_X, THREAD_GROUP_SIZE_Y);

// Kernel config
#define KERNEL_SIZE             5
#define KERNEL_RANGE_MIN        (-KERNEL_SIZE / 2)
#define KERNEL_RANGE_MAX        (KERNEL_SIZE / 2)

// Shared memory config
DEFINE_LDS_CONFIG(s_SM, KERNEL_SIZE);

// Keep composition data in FP32. Path-traced HDR values can overflow or lose too much
// precision in FP16 before the final output is clamped.
DECLARE_LDS_ARRAY_2D(float4, g_RawColor, KERNEL_SIZE);
DECLARE_LDS_ARRAY_2D(float4, g_DenoisedColor, KERNEL_SIZE);

// Feature flags
#define FLAGS_RAW_SOURCE_BLIT           (1 << 0)
#define FLAGS_SCALE_SRC                 (1 << 1)
#define FLAGS_MODE_2_SIGNAL             (1 << 2)

// Debug flags
#define FLAGS_DEBUG                     (1 << 16)
#define FLAGS_DEBUG_MODE_MASK           (0xFF << 16)
#define FLAGS_DEBUG_CORRELATION_BIAS    (1 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_SKIP_SIGNAL         (2 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_DENOISER_OUTPUT     (3 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_SPECULAR_COLOR      (4 << 17 | FLAGS_DEBUG)
#define FLAGS_DEBUG_DIFFUSE_COLOR       (5 << 17 | FLAGS_DEBUG)

// Mode 1/2 signal
Texture2D<half4> InDenoisedSignal1 : register(t0); // Fused or specular denoiser output
Texture2D<half4> InAlbedo1 : register(t1);         // Fused or specular albedo

// Mode 2 signal
Texture2D<half4> InDenoisedSignal2 : register(t2); // Diffuse denoiser output
Texture2D<half4> InAlbedo2 : register(t3);         // Diffuse albedo

// Secondary buffers
Texture2D<half4> InSkipSignal : register(t4);
Texture2D<half4> InRawColor : register(t5);
Texture2D<half4> InColorBeforeParticles : register(t6);

RWTexture2D<half4> OutColor : register(u0);

SamplerState LinearSampler : register(s0);

cbuffer CB_Comp : register(b0)
{
    float4 DstTexSize;

    float CorrelationBias;
    uint Flags;

    float2 _Padding;
}

static const float MAX_SAFE_RADIANCE = 65000.0f;

// The previous shader could blend a large amount of the original noisy colour back
// into the result. Limiting this blend is the main black-speckle/noise reduction.
static const float MAX_RAW_REINTRODUCTION = 0.22f;

// Expands the denoised neighbourhood envelope slightly so valid fine detail is not
// crushed while isolated black pixels and fireflies are rejected.
static const float NEIGHBOURHOOD_EXPANSION = 0.30f;

bool IsSet(uint mask)
{
    return (Flags & mask) == mask;
}

uint GetDebugMode()
{
    return Flags & FLAGS_DEBUG_MODE_MASK;
}

float3 SanitizeColor(float3 color)
{
    if (!all(isfinite(color)))
        return 0.0f;

    // Negative radiance and FP16-overflowing values can turn into black or flashing
    // artefacts later in the pipeline.
    return min(max(color, 0.0f), MAX_SAFE_RADIANCE);
}

float3 SanitizeAlbedo(float3 albedo)
{
    if (!all(isfinite(albedo)))
        return 0.0f;

    // DLSSD inputs should normally be close to 0..1. Leave some headroom for games
    // that provide slightly overbright reflectance while blocking corrupt values.
    return clamp(albedo, 0.0f, 4.0f);
}

float SafeLuminance(float3 color)
{
    const float luminance = GetLuminance(SanitizeColor(color));
    return isfinite(luminance) ? max(luminance, 0.0f) : 0.0f;
}

// Correlates raw noisy input with denoised colour using a modified SSIM.
float GetRawColorSimilarity(const uint2 gtID)
{
    const int2 smCenter = int2(gtID) + s_SM_HaloOffset;

    float meanD = 0.0f;
    float meanR = 0.0f;
    float meanDD = 0.0f; // D^2
    float meanRR = 0.0f; // R^2
    float meanRD = 0.0f; // R*D

    static const float s_RcpSigma = 1.0f / 1.2f;
    float totalWeight = 0.0f;

    [unroll]
    for (int x1 = KERNEL_RANGE_MIN; x1 <= KERNEL_RANGE_MAX; x1++)
    {
        [unroll]
        for (int y1 = KERNEL_RANGE_MIN; y1 <= KERNEL_RANGE_MAX; y1++)
        {
            const int2 smID = smCenter + int2(x1, y1);
            const float weight = exp(-(Square(x1) + Square(y1)) * s_RcpSigma);

            totalWeight += weight;

            const float lumD = g_DenoisedColor[smID.x][smID.y].a;
            const float lumR = g_RawColor[smID.x][smID.y].a;

            meanD += weight * lumD;
            meanDD += weight * Square(lumD);

            meanR += weight * lumR;
            meanRR += weight * Square(lumR);

            meanRD += weight * lumR * lumD;
        }
    }

    const float rcpTotalWeight = rcp(max(totalWeight, 1e-6f));

    meanD *= rcpTotalWeight;
    meanR *= rcpTotalWeight;
    meanDD *= rcpTotalWeight;
    meanRR *= rcpTotalWeight;
    meanRD *= rcpTotalWeight;

    const float meanDSq = Square(meanD);
    const float meanRSq = Square(meanR);

    // Variances: E[X^2] - E[X]^2.
    const float varD = max(meanDD - meanDSq, 0.0f);
    const float varR = max(meanRR - meanRSq, 2e-3f);

    const float devD = sqrt(varD);
    const float devR = sqrt(varR);

    // Covariance: E[RD] - E[R]E[D].
    const float covRD = meanRD - meanD * meanR;

    static const float s_SSIMStrictness = 1.0f;
    static const float s_COVThreshold = 0.2f;

    const float c1 = Square(1e-2f * s_SSIMStrictness);
    const float c2 = Square(3e-2f * s_SSIMStrictness);
    const float c3 = c2;

    const float strucCorrelation = (covRD + c3) * rcp(max(devD * devR + c3, 1e-6f));
    const float conCorrelation =
        (2.0f * devD * devR + c2) * rcp(max(varD + varR + c2, 1e-6f));
    const float lumCorrelation =
        (2.0f * meanD * meanR) * rcp(max(meanDSq + meanRSq + c1, 1e-6f));

    const float ssim = strucCorrelation * conCorrelation * lumCorrelation;

    // If the denoiser reports a flat area, do not reintroduce noisy raw variance.
    const float covD = devD * rcp(max(meanD, 1e-2f));

    // A large mean-luminance mismatch is usually an outlier, invalid history, a
    // disocclusion, or an exposure mismatch—not useful detail.
    const float meanDifference =
        abs(meanR - meanD) * rcp(max(max(meanR, meanD), 2e-2f));
    const float meanAgreement = 1.0f - smoothstep(0.18f, 0.70f, meanDifference);

    const float structuralAgreement = smoothstep(0.0f, 0.5f, ssim);
    const float detailConfidence = smoothstep(0.0f, s_COVThreshold, covD);

    return saturate(structuralAgreement * detailConfidence * meanAgreement);
}

float3 ClampRawToDenoisedNeighbourhood(const uint2 gtID, float3 rawColor)
{
    const int2 smCenter = int2(gtID) + s_SM_HaloOffset;

    float3 minimumColor = float3(MAX_SAFE_RADIANCE, MAX_SAFE_RADIANCE, MAX_SAFE_RADIANCE);
    float3 maximumColor = 0.0f;

    // A small neighbourhood clamp is enough to reject isolated black pixels and
    // fireflies without turning the image into a large spatial blur.
    [unroll]
    for (int y = -1; y <= 1; y++)
    {
        [unroll]
        for (int x = -1; x <= 1; x++)
        {
            const float3 sampleColor =
                SanitizeColor(g_DenoisedColor[smCenter.x + x][smCenter.y + y].rgb);

            minimumColor = min(minimumColor, sampleColor);
            maximumColor = max(maximumColor, sampleColor);
        }
    }

    const float3 range = maximumColor - minimumColor;

    // Scale the expansion with local HDR intensity, but retain a small absolute
    // allowance in dark regions.
    const float3 absoluteAllowance =
        max(maximumColor * 0.015f, float3(0.002f, 0.002f, 0.002f));
    const float3 padding = range * NEIGHBOURHOOD_EXPANSION + absoluteAllowance;

    return clamp(SanitizeColor(rawColor),
                 max(minimumColor - padding, 0.0f),
                 maximumColor + padding);
}

void PopulateSharedMemory(const uint2 groupID, const int2 gtID, const bool rawBlit)
{
    const int2 pxOrigin = int2(groupID * s_ThreadGroupSize) - s_SM_HaloOffset;
    const uint flatID = gtID.x + gtID.y * s_ThreadGroupSize.x;
    const int2 maxBounds = int2(DstTexSize.xy) - 1;

    [unroll]
    for (int i = 0; i < s_SM_LoadsPerThread; i++)
    {
        const uint smFlatID = flatID + i * NUM_THREADS;

        if (smFlatID < s_SM_ElementCount)
        {
            // Raw-blit path still needs the group barrier, but it doesn't read LDS.
            [branch]
            if (rawBlit)
                continue;

            const int2 smID = int2(smFlatID % s_SM_Size.x, smFlatID / s_SM_Size.x);
            const int2 px = clamp(pxOrigin + smID, int2(0, 0), maxBounds);

            float3 denoisedColor;
            float3 totalAlbedo;

            [branch]
            if (IsSet(FLAGS_MODE_2_SIGNAL))
            {
                const float3 denoisedSpecColor = SanitizeColor(InDenoisedSignal1[px].rgb);
                const float3 denoisedDiffColor = SanitizeColor(InDenoisedSignal2[px].rgb);
                const float3 specReflectance = SanitizeAlbedo(InAlbedo1[px].rgb);
                const float3 diffAlbedo = SanitizeAlbedo(InAlbedo2[px].rgb);

                totalAlbedo = specReflectance + diffAlbedo;
                denoisedColor =
                    denoisedSpecColor * specReflectance +
                    denoisedDiffColor * diffAlbedo;
            }
            else
            {
                totalAlbedo = SanitizeAlbedo(InAlbedo1[px].rgb);
                denoisedColor = SanitizeColor(InDenoisedSignal1[px].rgb) * totalAlbedo;
            }

            denoisedColor = SanitizeColor(
                denoisedColor + SanitizeColor(InSkipSignal[px].rgb));

            const float3 rawColor = SanitizeColor(InRawColor[px]);

            // Use demodulated colour for luma references, but modulated colour for RGB.
            const float3 rcpTotalAlbedo = rcp(max(totalAlbedo, 1e-3f));
            const float rawRef = SafeLuminance(rawColor * rcpTotalAlbedo);
            const float denoisedRef = SafeLuminance(denoisedColor * rcpTotalAlbedo);

            g_RawColor[smID.x][smID.y] = float4(rawColor, rawRef);
            g_DenoisedColor[smID.x][smID.y] = float4(denoisedColor, denoisedRef);
        }
    }

    GroupMemoryBarrierWithGroupSync();
}

[RootSignature(MainRS)]
[numthreads(THREAD_GROUP_SIZE_X, THREAD_GROUP_SIZE_Y, 1)]
void CSMain(uint3 groupID : SV_GroupID, uint3 gtID : SV_GroupThreadID)
{
    const uint2 px = groupID.xy * s_ThreadGroupSize + gtID.xy;
    const float2 uv = (float2(px) + 0.5f) * DstTexSize.zw;
    const bool inBounds = px.x < DstTexSize.x && px.y < DstTexSize.y;

    // PopulateSharedMemory contains GroupMemoryBarrierWithGroupSync(), which every
    // thread in the group must reach. Call it before any per-thread early return.
    const bool rawBlit = IsSet(FLAGS_RAW_SOURCE_BLIT);
    PopulateSharedMemory(groupID.xy, gtID.xy, rawBlit);

    [branch]
    if (rawBlit)
    {
        // Do not write to an out-of-bounds UAV coordinate.
        if (!inBounds)
            return;

        float4 sourceColor;

        [branch]
        if (IsSet(FLAGS_SCALE_SRC))
            sourceColor = InDenoisedSignal1.SampleLevel(LinearSampler, uv, 0);
        else
            sourceColor = InDenoisedSignal1[px];

        OutColor[px] =
            (half4)GetSafeFP16(float4(SanitizeColor(sourceColor.rgb), 1.0f));
        return;
    }

    const int2 smID = int2(gtID.xy) + s_SM_HaloOffset;

    if (!inBounds)
        return;

    // Estimate whether a small amount of raw detail is trustworthy.
    float rawBlendWeight =
        GetRawColorSimilarity(gtID.xy) * max(CorrelationBias, 0.0f);

    const float3 denoisedColor = SanitizeColor(g_DenoisedColor[smID.x][smID.y].rgb);
    const float3 rawColor = SanitizeColor(g_RawColor[smID.x][smID.y].rgb);
    const float3 clampedRawColor =
        ClampRawToDenoisedNeighbourhood(gtID.xy, rawColor);

    const float denoisedLum = SafeLuminance(denoisedColor);
    const float rawLum = SafeLuminance(rawColor);

    // Prevent a black raw outlier from replacing a stable denoised reflection.
    const float luminanceDifference =
        abs(rawLum - denoisedLum) * rcp(max(max(rawLum, denoisedLum), 2e-2f));
    const float perPixelAgreement =
        1.0f - smoothstep(0.16f, 0.65f, luminanceDifference);

    // Extra rejection specifically for dark/black speckles.
    const float darkDeficit =
        max(denoisedLum - rawLum, 0.0f) * rcp(max(denoisedLum, 2e-2f));
    const float blackOutlierRejection =
        1.0f - smoothstep(0.35f, 0.85f, darkDeficit);

    rawBlendWeight *= perPixelAgreement * blackOutlierRejection;
    rawBlendWeight = min(saturate(rawBlendWeight), MAX_RAW_REINTRODUCTION);

    [branch]
    if (IsSet(FLAGS_DEBUG))
    {
        switch (GetDebugMode())
        {
            case FLAGS_DEBUG_CORRELATION_BIAS:
                OutColor[px] = half4(TurboColormap(rawBlendWeight), 1.0f);
                break;

            case FLAGS_DEBUG_SKIP_SIGNAL:
                OutColor[px] =
                    (half4)GetSafeFP16(float4(SanitizeColor(InSkipSignal[px].rgb), 1.0f));
                break;

            case FLAGS_DEBUG_DENOISER_OUTPUT:
                OutColor[px] =
                    (half4)GetSafeFP16(float4(denoisedColor, 1.0f));
                break;

            case FLAGS_DEBUG_SPECULAR_COLOR:
                OutColor[px] = (half4)GetSafeFP16(float4(
                    SanitizeColor(InDenoisedSignal1[px].rgb) *
                    SanitizeAlbedo(InAlbedo1[px].rgb),
                    1.0f));
                break;

            case FLAGS_DEBUG_DIFFUSE_COLOR:
                OutColor[px] = (half4)GetSafeFP16(float4(
                    SanitizeColor(InDenoisedSignal2[px].rgb) *
                    SanitizeAlbedo(InAlbedo2[px].rgb),
                    1.0f));
                break;

            default:
                if (IsSet(FLAGS_MODE_2_SIGNAL))
                {
                    OutColor[px] = (half4)GetSafeFP16(float4(
                        SanitizeColor(InDenoisedSignal1[px].rgb) +
                        SanitizeColor(InDenoisedSignal2[px].rgb),
                        1.0f));
                }
                else
                {
                    OutColor[px] = (half4)GetSafeFP16(float4(
                        SanitizeColor(InDenoisedSignal1[px].rgb),
                        1.0f));
                }
                break;
        }

        return;
    }

    const float3 outColor =
        SanitizeColor(lerp(denoisedColor, clampedRawColor, rawBlendWeight));

    OutColor[px] =
        (half4)GetSafeFP16(float4(outColor, 1.0f));
}
