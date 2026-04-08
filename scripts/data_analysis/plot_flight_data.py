# File: plot_flight_data.py
# 依赖库: pip install pandas matplotlib numpy seaborn scipy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
import os
import seaborn as sns

# ================= 1. 全局学术风格设置 (Academic Style) =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] =['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'  # 紧凑保存，自动包容被移到外侧的图例

# 状态机颜色映射
STATE_COLORS = {
    'SEARCHING': '#eef2f5',  # 淡灰
    'ALIGNING': '#e6f2ff',   # 淡蓝
    'DESCENDING': '#e8f5e9'  # 淡绿
}

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['RunMode'] == 'CONTROL'].copy()
    start_time = df['TimeMs'].iloc[0]
    df['Time_s'] = (df['TimeMs'] - start_time) / 1000.0
    
    # 计算误差 L2 范数 (欧氏距离)
    df['Error_Norm'] = np.sqrt(df['ErrX']**2 + df['ErrY']**2)
    
    # 优美的 Savitzky-Golay 平滑滤波，去除高频毛刺，尽显高级感
    window = min(15, len(df) if len(df) % 2 != 0 else len(df) - 1)
    df['ErrX_smooth'] = savgol_filter(df['ErrX'], window, 3)
    df['ErrY_smooth'] = savgol_filter(df['ErrY'], window, 3)
    df['Error_Norm_smooth'] = savgol_filter(df['Error_Norm'], window, 3)
    return df

def draw_state_backgrounds(ax, df, hide_labels=False):
    """绘制状态机背景颜色带"""
    df['StateChange'] = df['TrackingState'] != df['TrackingState'].shift(1)
    change_indices = df[df['StateChange']].index.tolist()
    change_indices.append(df.index[-1] + 1)
    
    seen_labels = set()
    for i in range(len(change_indices) - 1):
        start_idx = change_indices[i]
        end_idx = change_indices[i+1] - 1
        state = df.loc[start_idx, 'TrackingState']
        start_time = df.loc[start_idx, 'Time_s']
        end_time = df.loc[end_idx, 'Time_s']
        
        label = state if (state not in seen_labels and not hide_labels) else ""
        seen_labels.add(state)
        ax.axvspan(start_time, end_time, color=STATE_COLORS.get(state, '#ffffff'), alpha=0.8, label=label)
        if i > 0:
            ax.axvline(start_time, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

# ================= 绘图函数区 =================

def plot_dataset_distribution(output_dir):
    """图1：动态扩展坐标轴的柱状图"""
    classes =['yellow_triangle', 'red_circle_1', 'purple_square', 'green_circle_3', 'blue_circle_2', 'land_h']
    train =[110, 126, 126, 131, 137, 135]
    val =[39, 36, 35, 33, 34, 37]
    test =[21, 17, 20, 21, 20, 19]
    totals =[t + v + te for t, v, te in zip(train, val, test)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid", font="Times New Roman")
    
    y = np.arange(len(classes))
    height = 0.6
    
    ax.barh(y, train, height, label='Train', color='#5DADE2')
    ax.barh(y, val, height, left=train, label='Validation', color='#7DCEA0')
    ax.barh(y, test, height, left=np.array(train)+np.array(val), label='Test', color='#AF7AC5')
    
    ax.set_xlim(0, max(totals) * 1.25)
    
    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('Number of annotated instances', fontsize=12)
    ax.set_title('Class Distribution of the Detection Dataset', fontsize=14, pad=15)
    
    ax.legend(loc='lower right', frameon=True, fontsize=11)
    
    for i in range(len(classes)):
        ax.text(totals[i] + 3, i, str(totals[i]), va='center', fontsize=11)
        
    sns.despine(left=True, bottom=True)
    plt.savefig(f'{output_dir}/Fig1_Dataset_Distribution.pdf')
    plt.savefig(f'{output_dir}/Fig1_Dataset_Distribution.png')
    plt.close()

def plot_error_convergence(df, output_dir):
    """图2：【彻底修复遮挡】双子图 - 视觉误差收敛与整体能量衰减"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 【修复1】：加大子图间距，防止下面的标题和上面的 X 轴重叠
    fig.subplots_adjust(hspace=0.25)
    
    # ----- Top Panel -----
    draw_state_backgrounds(ax1, df, hide_labels=True) 
    ax1.axhspan(-0.10, 0.10, color='#e0f2fe', alpha=0.4, label='Alignment band |e| $\leq$ 0.10')
    ax1.axhspan(-0.20, 0.20, color='#f0fdf4', alpha=0.3, label='Relaxed descent band |e| $\leq$ 0.20')
    ax1.axhline(0, color='black', linewidth=0.8)
    
    ax1.plot(df['Time_s'], df['ErrX_smooth'], label='ErrX (smoothed)', color='#1f77b4', linewidth=2.5)
    ax1.plot(df['Time_s'], df['ErrY_smooth'], label='ErrY (smoothed)', color='#d62728', linewidth=2.5)
    
    # 【修复2】：强制拉高 Y 轴上限，给顶部的标题和注释留出充足呼吸空间
    ax1.set_ylim(-0.9, 0.45)
    
    align_entries = df[(df['Error_Norm'] <= 0.10) & (df['TrackingState'] == 'ALIGNING')]
    if not align_entries.empty:
        t_align = align_entries.iloc[0]['Time_s']
        err_align = align_entries.iloc[0]['ErrX_smooth']
        # 【修复3】：将注释框向左平移 (-8)，向上平移 (+0.25)，彻底错开正中间的标题
        ax1.annotate(f'First entry into alignment band\nt = {t_align:.2f} s', 
                     xy=(t_align, err_align), xytext=(t_align-8, err_align+0.25),
                     arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    
    ax1.set_ylabel('Normalized Error', fontsize=12)
    ax1.set_title('Visual Error Convergence under Closed-loop Control', fontsize=14, pad=15)
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # ----- Bottom Panel -----
    draw_state_backgrounds(ax2, df, hide_labels=False)
    ax2.plot(df['Time_s'], df['Error_Norm_smooth'], label='$\|e\|_2$ (smoothed)', color='#6a1b9a', linewidth=2.5)
    
    ax2.axhline(0.10 * np.sqrt(2), color='#1f77b4', linestyle='--', linewidth=1, alpha=0.8, label='$\sqrt{2} \\times 0.10$')
    ax2.axhline(0.20 * np.sqrt(2), color='#2ca02c', linestyle='--', linewidth=1, alpha=0.8, label='$\sqrt{2} \\times 0.20$')
    
    ax2.set_ylim(0, 1.2) # 给范数图也保留上方间距
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Error norm $\|e\|_2$', fontsize=12)
    
    # 【修复4】：增加 pad 进一步推开标题与上图边界的距离
    ax2.set_title('Overall Error-Energy Decay with State Evolution', fontsize=14, pad=15)
    
    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(f'{output_dir}/Fig2_Visual_Error_Convergence.pdf')
    plt.savefig(f'{output_dir}/Fig2_Visual_Error_Convergence.png')
    plt.close()

def plot_altitude_profile(df, output_dir):
    """图3：三段式宽幅子图"""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.5, 4, 4])
    gs.update(hspace=0.35)
    
    ax_state = plt.subplot(gs[0])
    ax_main = plt.subplot(gs[1])
    ax_zoom = plt.subplot(gs[2])
    
    # ----- Panel 1: State Machine Bar -----
    draw_state_backgrounds(ax_state, df, hide_labels=True)
    ax_state.set_yticks([])
    ax_state.set_xticks([])
    for spine in ax_state.spines.values():
        spine.set_visible(False)
    ax_state.set_ylabel('State', fontsize=12)
    ax_state.set_title('Altitude Profile and State-Machine Evolution', fontsize=16, pad=15)
    
    for state in['SEARCHING', 'ALIGNING', 'DESCENDING']:
        state_data = df[df['TrackingState'] == state]
        if not state_data.empty:
            mid_time = state_data['Time_s'].mean()
            ax_state.text(mid_time, 0.5, state, ha='center', va='center', fontsize=11, fontweight='bold')
            
    # ----- Panel 2: Full Altitude & Command -----
    draw_state_backgrounds(ax_main, df, hide_labels=True)
    line1 = ax_main.plot(df['Time_s'], df['Height_m'], color='#2e7d32', linewidth=2.5, label='Ultrasonic height')
    line_th = ax_main.axhline(2.0, color='#e74c3c', linestyle='--', linewidth=1.5, label='2.0 m threshold')
    ax_main.set_ylabel('Height (m)', fontsize=12)
    ax_main.set_ylim(0, max(df['Height_m'].max() * 1.1, 22))
    
    ax_main_right = ax_main.twinx()
    line2 = ax_main_right.plot(df['Time_s'], df['CmdVert'], color='#1976d2', linewidth=2, label='CmdVert')
    ax_main_right.set_ylabel('Vertical cmd (m/s)', fontsize=12)
    ax_main_right.set_ylim(-0.6, 0.1)
    
    near_field_df = df[df['Height_m'] < 6.0]
    x1 = near_field_df['Time_s'].min() - 0.5 if not near_field_df.empty else 18.0
    x2 = df['Time_s'].max()
    ax_main.axvspan(x1, x2, color='gray', alpha=0.15, hatch='///', label='Zoomed Near-field Area')
    
    lines = line1 + [line_th] + line2
    labels =[l.get_label() for l in lines] + ['Zoomed Near-field Area']
    ax_main.legend(lines +[plt.Rectangle((0,0),1,1, fc="gray", alpha=0.15, hatch='///')], 
                   labels, loc='center left', bbox_to_anchor=(1.08, 0.5))
    ax_main.grid(True, linestyle='--', alpha=0.5)

    # ----- Panel 3: Zoomed Altitude (Near Field) -----
    draw_state_backgrounds(ax_zoom, df, hide_labels=True)
    line3 = ax_zoom.plot(df['Time_s'], df['Height_m'], color='#2e7d32', linewidth=3.0, label='Ultrasonic height')
    line_th2 = ax_zoom.axhline(2.0, color='#e74c3c', linestyle='--', linewidth=2.0, label='2.0 m threshold')
    
    ax_zoom.set_xlim(x1, x2)
    ax_zoom.set_ylim(1.5, 6.0)
    ax_zoom.set_xlabel('Time (s)', fontsize=13)
    ax_zoom.set_ylabel('Height (m)', fontsize=12)
    ax_zoom.set_title('Near-field Descent Zoom (Height < 6.0m)', fontsize=13, pad=10)
    
    ax_zoom_right = ax_zoom.twinx()
    line4 = ax_zoom_right.plot(df['Time_s'], df['CmdVert'], color='#1976d2', linewidth=2.5, label='CmdVert')
    ax_zoom_right.set_ylabel('Vertical cmd (m/s)', fontsize=12)
    ax_zoom_right.set_ylim(-0.6, 0.1)
    
    lines_z = line3 + [line_th2] + line4
    labels_z = [l.get_label() for l in lines_z]
    ax_zoom.legend(lines_z, labels_z, loc='center left', bbox_to_anchor=(1.08, 0.5))
    ax_zoom.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(f'{output_dir}/Fig3_Altitude_State_Evolution.pdf')
    plt.savefig(f'{output_dir}/Fig3_Altitude_State_Evolution.png')
    plt.close()

def plot_control_smoothing(df, output_dir):
    """图4：指数衰减与伪高频平滑效果展示"""
    plt.figure(figsize=(10, 5))
    
    mask = (df['Time_s'] >= 1.0) & (df['Time_s'] <= 6.0)
    df_zoom = df[mask]
    
    plt.plot(df_zoom['Time_s'], df_zoom['CmdRoll'], color='#8e44ad', linewidth=2.5, label='CmdRoll (Smoothed)')
    plt.plot(df_zoom['Time_s'], df_zoom['CmdPitch'], color='#d35400', linewidth=2.5, label='CmdPitch (Smoothed)')
    plt.scatter(df_zoom['Time_s'], df_zoom['CmdRoll'], color='#8e44ad', s=20, alpha=0.5)
    plt.scatter(df_zoom['Time_s'], df_zoom['CmdPitch'], color='#d35400', s=20, alpha=0.5)

    plt.title('High-Frequency Exponential Decay Control Output (Zoomed)', fontsize=14, pad=15)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Command Output', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    plt.savefig(f'{output_dir}/Fig4_Control_Command_Smoothing.pdf')
    plt.savefig(f'{output_dir}/Fig4_Control_Command_Smoothing.png')
    plt.close()

def plot_phase_portrait(df, output_dir):
    """图5：2D 误差轨迹图 (Phase Portrait)"""
    plt.figure(figsize=(7, 7))
    
    c1 = plt.Circle((0, 0), 0.10, color='#1f77b4', fill=True, alpha=0.1, label='Alignment Band (0.10)')
    c2 = plt.Circle((0, 0), 0.20, color='#2ca02c', fill=True, alpha=0.05, label='Relaxed Band (0.20)')
    plt.gca().add_patch(c1)
    plt.gca().add_patch(c2)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    
    sc = plt.scatter(df['ErrX'], df['ErrY'], c=df['Time_s'], cmap='viridis', s=25, alpha=0.8, edgecolor='none')
    plt.scatter(df['ErrX'].iloc[0], df['ErrY'].iloc[0], color='red', marker='s', s=100, label='Start Position', zorder=5)
    plt.scatter(df['ErrX'].iloc[-1], df['ErrY'].iloc[-1], color='blue', marker='*', s=200, label='Final Position', zorder=5)
    
    cbar = plt.colorbar(sc, shrink=0.8)
    cbar.set_label('Time (s)', fontsize=11)
    
    plt.title('2D Image-Plane Error Trajectory', fontsize=14, pad=15)
    plt.xlabel('Normalized Error X', fontsize=12)
    plt.ylabel('Normalized Error Y', fontsize=12)
    
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5)) 
    
    plt.savefig(f'{output_dir}/Fig5_Error_Phase_Portrait.pdf')
    plt.savefig(f'{output_dir}/Fig5_Error_Phase_Portrait.png')
    plt.close()

def plot_3d_trajectory(df, output_dir):
    """图6：3D 空中下降轨迹图"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(df['ErrX_smooth'], df['ErrY_smooth'], df['Height_m'], 
                    c=df['Time_s'], cmap='coolwarm', s=15, alpha=0.8)
    ax.plot(df['ErrX_smooth'], df['ErrY_smooth'], df['Height_m'], color='gray', alpha=0.4, linewidth=1)
    
    ax.scatter(df['ErrX_smooth'].iloc[0], df['ErrY_smooth'].iloc[0], df['Height_m'].iloc[0], 
               color='red', marker='s', s=100, label='Start (Hover)')
    ax.scatter(df['ErrX_smooth'].iloc[-1], df['ErrY_smooth'].iloc[-1], df['Height_m'].iloc[-1], 
               color='green', marker='*', s=200, label='Handover (< 2m)')
    
    ax.set_xlabel('Normalized Error X', labelpad=10)
    ax.set_ylabel('Normalized Error Y', labelpad=10)
    ax.set_zlabel('Altitude (m)', labelpad=10)
    ax.set_title('3D Spatial Descent Trajectory Approximation', fontsize=15, pad=20)
    
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Flight Time (s)')
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    
    plt.savefig(f'{output_dir}/Fig6_3D_Trajectory.pdf')
    plt.savefig(f'{output_dir}/Fig6_3D_Trajectory.png')
    plt.close()

def plot_control_effort_boxplot(df, output_dir):
    """图7：控制律输出箱线图"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    sns.set_theme(style="whitegrid", font="Times New Roman")
    
    states_order =['SEARCHING', 'ALIGNING', 'DESCENDING']
    
    sns.boxplot(x='TrackingState', y='CmdRoll', data=df, ax=axes[0], palette='pastel', order=states_order)
    axes[0].set_title('Roll Command Distribution', fontsize=13)
    axes[0].set_ylabel('CmdRoll Output', fontsize=12)
    axes[0].set_xlabel('')
    
    sns.boxplot(x='TrackingState', y='CmdPitch', data=df, ax=axes[1], palette='pastel', order=states_order)
    axes[1].set_title('Pitch Command Distribution', fontsize=13)
    axes[1].set_ylabel('CmdPitch Output', fontsize=12)
    axes[1].set_xlabel('Flight Phase', fontsize=12)
    
    sns.boxplot(x='TrackingState', y='CmdVert', data=df, ax=axes[2], palette='pastel', order=states_order)
    axes[2].set_title('Vertical Command Distribution', fontsize=13)
    axes[2].set_ylabel('CmdVert Output', fontsize=12)
    axes[2].set_xlabel('')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Fig7_Control_Effort.pdf')
    plt.savefig(f'{output_dir}/Fig7_Control_Effort.png')
    plt.close()

if __name__ == "__main__":
    target_csv = "FlightData_20260403_163531.csv"
    output_directory = "plots"
    os.makedirs(output_directory, exist_ok=True)
    
    plot_dataset_distribution(output_directory)
    print("1/7 Generated: Fig 1 Dataset Distribution")
    
    if os.path.exists(target_csv):
        print(f"\nReading flight data from: {target_csv}")
        df_flight = load_and_preprocess(target_csv)
        
        plot_error_convergence(df_flight, output_directory)
        print("2/7 Generated: Fig 2 Visual Error Convergence (Title Occlusion Fixed!)")
        
        plot_altitude_profile(df_flight, output_directory)
        print("3/7 Generated: Fig 3 Altitude & State Evolution")
        
        plot_control_smoothing(df_flight, output_directory)
        print("4/7 Generated: Fig 4 Control Smoothing Zoom")
        
        plot_phase_portrait(df_flight, output_directory)
        print("5/7 Generated: Fig 5 2D Error Phase Portrait")
        
        plot_3d_trajectory(df_flight, output_directory)
        print("6/7 Generated: Fig 6 3D Spatial Trajectory")
        
        plot_control_effort_boxplot(df_flight, output_directory)
        print("7/7 Generated: Fig 7 Control Effort Boxplots")
        
        print(f"\n✅ All 7 academic figures have been saved successfully to '{os.path.abspath(output_directory)}'!")
    else:
        print(f"Error: Cannot find file '{target_csv}'. Please ensure the file is in the same directory as this script.")