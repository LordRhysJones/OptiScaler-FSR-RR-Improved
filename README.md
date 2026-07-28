<div align="center">

![Logo](https://github.com/user-attachments/assets/c7dad5da-0b29-4710-8a57-b58e4e407abd)

# OptiScaler FSR Ray Reconstruction Edition

### Enhanced FSR Ray Reconstruction for DLSS-Compatible Games

*A community-driven fork focused on improving FSR Ray Reconstruction quality, reducing visual artifacts, and expanding compatibility.*

</div>

---

## Overview

OptiScaler FSR Ray Reconstruction Edition is a specialized fork of the original **OptiScaler** project, with development focused primarily on **AMD's FSR Ray Reconstruction (FSR RR)**.

The goal of this project isn't simply to expose FSR Ray Reconstruction in supported games—it is to improve the overall quality of the reconstruction process itself.

Since the project began, we've spent considerable time refining the implementation to reduce common visual issues including instability, artifacting, flickering, and inconsistent reconstruction quality.

Our primary development platform has been **Cyberpunk 2077**, where these improvements have produced a substantial increase in image quality compared to the original OptiScaler FSR RR implementation.

Although Cyberpunk remains our main testing title, we're continuously working to improve compatibility across as many supported games as possible.

---

# Features

This fork introduces numerous improvements over the original OptiScaler FSR RR implementation, including:

- Improved FSR Ray Reconstruction quality
- Reduced visual artifacting
- Better temporal stability
- Cleaner reconstruction during camera movement
- Improved handling of noisy reflections
- Better reconstruction consistency
- General quality improvements throughout the rendering pipeline
- Ongoing compatibility improvements

The project continues to evolve as additional games are tested and new issues are identified.

---

# Current Development Focus

Our primary focus is quality.

Rather than chasing benchmark numbers, this fork aims to produce a cleaner and more stable image while maintaining the excellent flexibility provided by OptiScaler.

Current development priorities include:

- Improving reconstruction quality
- Reducing visual artifacts
- Increasing compatibility
- Fixing game-specific issues
- Improving stability across different rendering implementations

---

# Tested Games

## ✅ Cyberpunk 2077

Cyberpunk 2077 is currently the primary development and testing platform.

Compared to previous FSR Ray Reconstruction implementations, users should generally notice:

- Reduced shimmering
- Cleaner reflections
- Improved image stability
- Reduced ghosting
- Better temporal reconstruction
- Fewer distracting artifacts

These improvements represent the main goal of this project.

You can view a full compatibility list of different games that are actively tested, untested or not working here:

---

# Compatibility

This project is intended to work anywhere OptiScaler's FSR Ray Reconstruction implementation is supported.

However, compatibility depends heavily on how individual games implement:

- DLSS
- Ray Tracing
- Motion Vectors
- Depth Buffers
- Frame Generation
- Rendering Pipelines

Because every game integrates these systems differently, results will vary.

Some games may work flawlessly.

Others may require additional tweaks.

Some titles may not currently function correctly.

If a game hasn't been tested yet, we'd love to hear your feedback.

---

# Performance

One interesting side effect we've observed during development is that some users may see a small performance improvement.

This isn't the primary objective of the project, but improvements to reconstruction can occasionally reduce GPU overhead depending on the game and hardware configuration.

Performance is influenced by many factors including:

- GPU model
- Driver version
- Game engine
- Graphics settings
- Resolution
- Ray Tracing workload

Because of this, we **cannot guarantee an FPS increase** on every system.

Some users may see higher frame rates.

Others may experience identical performance with improved image quality.

Your experience will depend on your hardware and the specific game.

---

# Installation

Installation is identical to the original OptiScaler project.

Simply follow the normal OptiScaler installation instructions but using our version, which can be downloaded from the releases tab or by clicking this link: https://github.com/LordRhysJones/OptiScaler-FSR-RR-Improved/releases/download/Beta/OptiScaler-FSR-RR-Improved-V0.1B.zip

---

# Community Testing

This project benefits enormously from community feedback.

If you test a game that hasn't been verified yet, please consider reporting:

- Whether it works
- Any visual issues encountered
- Performance observations
- Screenshots or comparison images
- Recommended settings

Every compatibility report helps improve future releases.

---

# Roadmap

Development is ongoing.

Some of the areas we're actively working on include:

- Wider game compatibility
- Additional artifact reduction
- Improved reconstruction stability
- Better image consistency
- Further optimization
- Game-specific fixes

---

# Credits

This project is built upon the incredible work of the original OptiScaler developers.

Huge thanks to everyone who contributed to the original project, especially:

- OptiScaler Team
- PotatoOfDoom (CyberFSR2)
- Artur
- LukeFZ
- Nukem

Without their work, this project would not exist.

---

# Disclaimer

This is an unofficial community fork of OptiScaler focused specifically on improving FSR Ray Reconstruction.

While we strive for broad compatibility, not every game implements rendering technologies in the same way. As a result, compatibility and performance may vary between titles.

Our goal is simple:

Deliver the best possible FSR Ray Reconstruction experience while continually improving support across more games.

---

<div align="center">

### ⭐ Enjoy the project?

If this fork improves your experience, consider starring the repository and sharing compatibility reports with the community.

Every report helps make future releases even better.

</div>
