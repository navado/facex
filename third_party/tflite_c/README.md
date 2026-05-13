# Vendored TFLite C-API headers

Minimal subset of public TensorFlow Lite C-API headers needed to compile
`src/backend_tflite.c`. Avoids dragging in the full TF source tree or
requiring a `libtensorflow-lite-dev` package that NXP runtime images
(and many Yocto BSPs) do not ship.

- **Source:** https://github.com/tensorflow/tensorflow
- **Tag:** `v2.19.0` — matches the `libtensorflow-lite.so.2.19.0` shipped
  in the NXP i.MX walnascar (BSP 6.12) image and the upstream TFLite 2.19
  release commonly used with i.MX 93 (Ethos-U), i.MX 95 (eIQ Neutron N3),
  and i.MX 8M Plus (VxDelegate).
- **License:** Apache-2.0 (upstream).

## Files

```
tensorflow/builtin_ops.h
tensorflow/lite/c/{c_api, c_api_experimental, c_api_opaque, c_api_types, common}.h
tensorflow/lite/core/c/{c_api, c_api_experimental, c_api_opaque, c_api_types, common, operator}.h
tensorflow/lite/core/async/c/types.h
tensorflow/compiler/mlir/lite/core/c/tflite_types.h
```

Total: 14 files / ~280 KB. Refreshing for a newer TFLite minor version is
an iterative compile-and-fetch loop — compile `src/backend_tflite.c` with
`-Ithird_party/tflite_c/include`, fetch whichever header the first
`fatal error: <header>: No such file or directory` points at from the
matching TF release tag, repeat until clean.
