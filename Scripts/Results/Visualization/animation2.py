import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_field(ax):
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_title("NFL Field")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    plt.gca().invert_yaxis()  # Invert y-axis to match NFL field orientation

def random_team_colors():
    return np.random.rand(3,), np.random.rand(3,)

def plot_play_frame(ax, player_data, ball_data, los, togo_line, play_desc):
    h_team = player_data.loc[player_data['homeTeamFlag'] == 1, 'teamAbbr'].iloc[0]
    a_team = player_data.loc[player_data['homeTeamFlag'] == 0, 'teamAbbr'].iloc[0]

    h_team_color, a_team_color = random_team_colors()

    ax.axvline(x=los, color="#0d41e1", linestyle="--", label="Line of Scrimmage")
    ax.axvline(x=togo_line, color="#f9c80e", linestyle="--", label="1st Down Marker")

    ax.quiver(player_data.loc[player_data['teamAbbr'] == a_team, 'x'],
              player_data.loc[player_data['teamAbbr'] == a_team, 'y'],
              player_data.loc[player_data['teamAbbr'] == a_team, 'v_x'],
              player_data.loc[player_data['teamAbbr'] == a_team, 'v_y'],
              color=a_team_color, scale=5, label=f"{a_team} Velocities")

    ax.quiver(player_data.loc[player_data['teamAbbr'] == h_team, 'x'],
              player_data.loc[player_data['teamAbbr'] == h_team, 'y'],
              player_data.loc[player_data['teamAbbr'] == h_team, 'v_x'],
              player_data.loc[player_data['teamAbbr'] == h_team, 'v_y'],
              color=h_team_color, scale=5, label=f"{h_team} Velocities")

    ax.scatter(player_data.loc[player_data['teamAbbr'] == a_team, 'x'],
               player_data.loc[player_data['teamAbbr'] == a_team, 'y'],
               c=a_team_color, edgecolors='k', marker='o', label=f"{a_team} Players")

    ax.scatter(player_data.loc[player_data['teamAbbr'] == h_team, 'x'],
               player_data.loc[player_data['teamAbbr'] == h_team, 'y'],
               c=h_team_color, edgecolors='k', marker='o', label=f"{h_team} Players")

    ax.scatter(ball_data['x'], ball_data['y'], c='#935e38', edgecolors='#d9d9d9', marker='o', s=100, label="Ball")

    ax.scatter(player_data.loc[player_data['isFastestFlag'] == 1, 'x'],
               player_data.loc[player_data['isFastestFlag'] == 1, 'y'],
               c='#e9ff70', marker='o', s=150, alpha=0.5, label="Fastest Players")

    ax.text(0.5, 1.05, play_desc, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.5, -0.1, "Source: NFL Next Gen Stats", horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=8, color='gray')

    ax.legend()

def animate_tracking_data(play_data):
    play_desc = play_data['playDescription'].iloc[0]
    play_dir = play_data['playDirection'].iloc[0]
    yards_togo = play_data['yardsToGo'].iloc[0]
    los = play_data['absoluteYardlineNumber'].iloc[0]
    togo_line = los - yards_togo if play_dir == "left" else los + yards_togo

    player_data = play_data[play_data['displayName'] != "ball"]
    ball_data = play_data[play_data['displayName'] == "ball"]

    player_data['dir_rad'] = np.deg2rad(player_data['dir'])
    player_data['v_x'] = np.sin(player_data['dir_rad']) * player_data['s']
    player_data['v_y'] = np.cos(player_data['dir_rad']) * player_data['s']

    player_data['isFastestFlag'] = player_data.groupby(['frame', 'teamAbbr'])['s'].transform('max') == player_data['s']

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_field(ax)
    plot_play_frame(ax, player_data, ball_data, los, togo_line, play_desc)

    def update(frame):
        ax.clear()
        plot_field(ax)
        plot_play_frame(ax, player_data, ball_data, los, togo_line, play_desc, frame)

    ani = animation.FuncAnimation(fig, update, frames=range(int(player_data['frame'].min()), int(player_data['frame'].max()) + 1),
                                  interval=100, repeat=False)
    
    plt.show()



