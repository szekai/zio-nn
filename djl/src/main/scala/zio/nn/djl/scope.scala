package zio.nn.djl

import ai.djl.ndarray.NDManager
import zio.*

/** Resource management helpers for DJL.
  *
  * Usage:
  * {{{
  *   import zio.nn.djl.scope.*
  *   import zio.nn.TensorOps.*
  *
  *   withNDManager {
  *     for
  *       a <- create(data)
  *       b <- create(more)
  *       c <- add(a, b)
  *     yield c
  *   }
  *   // NDManager auto-closed by ZIO Scope
  * }}}
  */
object scope:

  /** Provide an NDManager as a ZIO Scope resource.
    * The manager is created at scope entry and closed at scope exit.
    */
  def withNDManager[R, E, A](zio: NDManager ?=> ZIO[R, E, A]): ZIO[R & Scope, E, A] =
    ZIO.acquireRelease(
      ZIO.attemptBlocking(NDManager.newBaseManager()).orDie
    )(ndm => ZIO.attemptBlocking(ndm.close()).orDie).flatMap { ndm =>
      given NDManager = ndm
      zio
    }
