package zio.nn

// Re-export DJL wrappers so consumers just write `import zio.nn.*`.
// When zio-nn-djl is on the classpath, these become available globally.
export zio.nn.djl.{ZModel, ZPredictor, ZTrainer, Backend, defaultConfig, mseLoss}
