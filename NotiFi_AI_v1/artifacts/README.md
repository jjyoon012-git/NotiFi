# Model Artifacts

- `notifi_ai_v1_core.ts`: CSI core, classification, risk, coarse pose, root
- `notifi_ai_v1_features.ts`: retrieval features and motion descriptors
- `notifi_ai_v1_core_cpu.ts`: CPU-compatible CSI core
- `notifi_ai_v1_features_cpu.ts`: CPU-compatible retrieval features
- `notifi_ai_v1_retrieval.pt`: train-only motion bank and retrieval networks
- `manifest.json`: artifact sizes, SHA-256 hashes, and source provenance

All five model files are required. The runtime selects CUDA or CPU TorchScript
automatically. Verify them with:

```powershell
python scripts\verify_release.py
```
