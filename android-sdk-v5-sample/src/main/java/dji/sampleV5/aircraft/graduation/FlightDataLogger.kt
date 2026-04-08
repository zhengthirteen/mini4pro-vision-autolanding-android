package dji.sampleV5.aircraft.graduation

import android.content.Context
import android.os.Environment
import android.util.Log
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object FlightDataLogger {
    private var writer: BufferedWriter? = null
    private var isLogging = false
    private var startTimeMs = 0L

    fun init(context: Context) {
        if (isLogging) return
        try {
            // 【终极保险】：双重路径降级策略
            // 首选：手机公共文档目录 (红米文件管理器直接可见，可微信直发)
            var logDir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), "FlightLogs")

            // 如果公共目录创建失败（可能因为 Android 11+ 权限收紧未授权）
            if (!logDir.exists() && !logDir.mkdirs()) {
                // 降级使用：App专属私有目录（免权限，插电脑 Android Studio 可见）
                logDir = File(context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "FlightLogs")
                if (!logDir.exists()) {
                    logDir.mkdirs()
                }
                Log.w("FlightDataLogger", "无法访问公共目录，已降级使用私有目录: ${logDir.absolutePath}")
            } else {
                Log.i("FlightDataLogger", "日志已启动，请在手机文件管理器 [文档/FlightLogs] 查找: ${logDir.absolutePath}")
            }

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val logFile = File(logDir, "FlightData_$timestamp.csv")

            // 使用 BufferedWriter 提升高频写入性能
            writer = BufferedWriter(FileWriter(logFile, true))

            // 写入 CSV 表头，新增 RunMode 字段区分控制和待机
            writer?.append("TimeMs,RunMode,TrackingState,SearchMode,Height_m,ErrX,ErrY,BoxRatio,CmdRoll,CmdPitch,CmdVert\n")
            writer?.flush()

            startTimeMs = System.currentTimeMillis()
            isLogging = true
        } catch (e: Exception) {
            Log.e("FlightDataLogger", "日志初始化失败: ${e.message}")
        }
    }

    // 高频写入单行数据
    fun log(
        runMode: String,
        state: String,
        searchMode: String,
        height: Double,
        errX: Double,
        errY: Double,
        boxRatio: Double,
        cmdRoll: Double,
        cmdPitch: Double,
        cmdVert: Double
    ) {
        if (!isLogging || writer == null) return
        try {
            val timeSinceStart = System.currentTimeMillis() - startTimeMs
            val line = String.format(
                Locale.US,
                "%d,%s,%s,%s,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                timeSinceStart, runMode, state, searchMode, height, errX, errY, boxRatio, cmdRoll, cmdPitch, cmdVert
            )
            writer?.append(line)

            // 【最关键的一步】：实时落盘！无论闪退还是炸机，只要执行到这里数据就在硬盘里！
            writer?.flush()
        } catch (e: Exception) {
            Log.e("FlightDataLogger", "写入日志失败: ${e.message}")
        }
    }

    fun close() {
        try {
            writer?.flush()
            writer?.close()
            writer = null
            isLogging = false
            Log.i("FlightDataLogger", "日志已安全关闭。")
        } catch (e: Exception) {
            Log.e("FlightDataLogger", "关闭日志失败: ${e.message}")
        }
    }
}