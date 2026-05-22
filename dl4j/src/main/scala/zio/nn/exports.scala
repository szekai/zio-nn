package zio.nn

// Re-export DL4J wrappers so consumers just write `import zio.nn.*`.
// When zio-nn-dl4j is on the classpath, these become available globally.
export zio.nn.dl4j.{ZModel, ZGraphModel, Backend, toDataSet}
