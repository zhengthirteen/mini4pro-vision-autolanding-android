package dji.sampleV5.aircraft.graduation

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

object AssetUtils {
    fun getAssetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        // 如果文件已存在且大小 > 0，直接返回路径（避免每次启动都拷贝，省时间）
        // 如果你更新了模型，记得卸载App重装，或者把这里逻辑改成强制覆盖
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }

        try {
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return file.absolutePath
    }
}