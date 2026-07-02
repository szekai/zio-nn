package zio.nn.djl

import ai.djl.Device

/** Device selection utilities for the DJL backend.
  *
  * DJL's [[Device]] class provides `cpu()` and `gpu()` but no `mps()` —
  * Metal Performance Shaders for Apple Silicon require `Device.of("mps", 0)`.
  *
  * ==Usage==
  * {{{
  *   import zio.nn.djl.DeviceHelper.*
  *
  *   // Explicit MPS (Apple Silicon)
  *   ZModel.create(arch, device = mps)
  *
  *   // Auto-detect: MPS on macOS aarch64, GPU if available, else CPU
  *   ZModel.create(arch, device = best())
  * }}}
  */
object DeviceHelper:

  /** A DJL [[Device]] targeting Metal Performance Shaders (Apple Silicon).
    *
    * Creates `Device.of("mps", 0)` — the device type string `"mps"` is
    * passed through to the underlying PyTorch engine. Whether MPS is actually
    * used depends on the PyTorch native build (the macOS aarch64 build does
    * include MPS support in recent versions).
    */
  val mps: Device = Device.of("mps", 0)

  /** Select the best available device at runtime.
    *
    * Detection order:
    *  1. macOS on aarch64 → returns [[mps]]
    *  2. Any platform → returns `Device.gpu()` if a GPU is available
    *  3. Fallback → returns `Device.cpu()`
    *
    * Note: DJL's `Device.gpu()` does not auto-detect MPS on Apple Silicon —
    * that is why the explicit macOS aarch64 check is needed.
    *
    * @return
    *   The best Device for the current system.
    */
  def best(): Device =
    val os   = System.getProperty("os.name", "").toLowerCase
    val arch = System.getProperty("os.arch", "").toLowerCase

    if os.contains("mac") && arch.contains("aarch64") then
      mps
    else
      try
        val gpu = Device.gpu()
        if gpu ne Device.cpu() then gpu else Device.cpu()
      catch
        case _: Exception => Device.cpu()
