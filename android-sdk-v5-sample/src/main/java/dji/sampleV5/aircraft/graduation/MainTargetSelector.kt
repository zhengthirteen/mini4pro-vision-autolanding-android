package dji.sampleV5.aircraft.graduation

import org.opencv.core.Rect
import kotlin.math.hypot
import kotlin.math.min

object MainTargetSelector {

    fun selectMainTarget(
        allTargets: List<OverlayTarget>,
        lastMainTarget: OverlayTarget?,
        imageWidth: Int,
        imageHeight: Int
    ): OverlayTarget? {
        if (allTargets.isEmpty()) return null

        val landHTargets = allTargets.filter { it.label == "land_h" }
        if (landHTargets.isNotEmpty()) {
            return if (lastMainTarget != null) {
                landHTargets.minByOrNull { rectCenterDistance(it.rect, lastMainTarget.rect) }
            } else {
                landHTargets.minByOrNull { distanceToImageCenter(it, imageWidth, imageHeight) }
            }
        }

        lastMainTarget?.let { previous ->
            val sameLabelTargets = allTargets.filter { it.label == previous.label }
            if (sameLabelTargets.isNotEmpty()) {
                val continuityThresholdPx = min(imageWidth, imageHeight) * 0.25
                val continuousTargets = sameLabelTargets.filter {
                    rectCenterDistance(it.rect, previous.rect) <= continuityThresholdPx || rectIou(it.rect, previous.rect) >= 0.10
                }
                if (continuousTargets.isNotEmpty()) {
                    return continuousTargets.minByOrNull { rectCenterDistance(it.rect, previous.rect) }
                }
            }
        }

        return allTargets.minByOrNull { distanceToImageCenter(it, imageWidth, imageHeight) }
    }

    private fun distanceToImageCenter(target: OverlayTarget, imageWidth: Int, imageHeight: Int): Double {
        val dx = target.centerX - imageWidth / 2.0
        val dy = target.centerY - imageHeight / 2.0
        return hypot(dx, dy)
    }

    private fun rectCenterDistance(a: Rect, b: Rect): Double {
        val ax = a.x + a.width / 2.0
        val ay = a.y + a.height / 2.0
        val bx = b.x + b.width / 2.0
        val by = b.y + b.height / 2.0
        return hypot(ax - bx, ay - by)
    }

    private fun rectIou(a: Rect, b: Rect): Double {
        val left = maxOf(a.x, b.x)
        val top = maxOf(a.y, b.y)
        val right = minOf(a.x + a.width, b.x + b.width)
        val bottom = minOf(a.y + a.height, b.y + b.height)

        if (right <= left || bottom <= top) return 0.0

        val inter = (right - left).toDouble() * (bottom - top).toDouble()
        val union = a.width.toDouble() * a.height.toDouble() +
            b.width.toDouble() * b.height.toDouble() - inter

        return if (union <= 0.0) 0.0 else inter / union
    }
}
