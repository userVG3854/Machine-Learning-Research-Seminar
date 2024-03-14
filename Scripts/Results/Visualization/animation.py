import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

def plot_field(ax):
    img = mpimg.imread('court.png')  # Load the image
    ax.imshow(img, extent=[-15, 135, -5, 60])  # Adjusted image extent
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_title("NFL Field")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    plt.gca().invert_yaxis()  # Invert y-axis to match NFL field orientation

def random_team_colors(teams):
    return {team: np.random.rand(3,) for team in teams}

def plot_play_frame(ax, player_data, ball_data, los, togo_line, play_desc, team_colors):
    teams = player_data['teamAbbr'].unique()

    ax.axvline(x=los, color="#0d41e1", linestyle="--", label="Line of Scrimmage")
    ax.axvline(x=togo_line, color="#f9c80e", linestyle="--", label="1st Down Marker")

    for team in teams:
        team_players = player_data[player_data['teamAbbr'] == team]
        ax.quiver(team_players['x'], team_players['y'], team_players['v_x'], team_players['v_y'],
                  color=team_colors[team], scale=40, label=f"{team} Velocities")  # Reduced arrow size

        ax.scatter(team_players['x'], team_players['y'], c=team_colors[team], edgecolors='k',
                   marker='o', label=f"{team} Players")

    ax.scatter(ball_data['x'], ball_data['y'], c='#935e38', edgecolors='#d9d9d9',
               marker='o', s=100, label="Ball")

    ax.text(0.5, 1.05, play_desc, horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.5, -0.1, "Source: NFL Next Gen Stats", horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=8, color='gray')

    ax.legend()

def update(frame, player_data, ball_data, los, togo_line, play_desc, team_colors, ax):
    ax.clear()
    plot_field(ax)
    frame_data = player_data[player_data['frame'] == frame]
    player_frame_data = frame_data[frame_data['displayName'] != "ball"]
    ball_frame_data = frame_data[frame_data['displayName'] == "ball"]
    plot_play_frame(ax, player_frame_data, ball_frame_data, los, togo_line, play_desc, team_colors)

def animate_tracking_data(play_data):
    play_desc = play_data['playDescription'].iloc[0]
    play_dir = play_data['playDirection_left'].iloc[0]
    yards_togo = play_data['yardsToGo'].iloc[0]
    los = play_data['absoluteYardlineNumber'].iloc[0]
    togo_line = los - yards_togo if play_dir == "left" else los + yards_togo

    player_data = play_data[play_data['displayName'] != "ball"]
    ball_data = play_data[play_data['displayName'] == "ball"]

    player_data['dir_rad'] = np.deg2rad(player_data['dir'])
    player_data['v_x'] = np.sin(player_data['dir_rad']) * player_data['s']
    player_data['v_y'] = np.cos(player_data['dir_rad']) * player_data['s']

    teams = player_data['teamAbbr'].unique()
    team_colors = random_team_colors(teams)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_field(ax)
    plot_play_frame(ax, player_data, ball_data, los, togo_line, play_desc, team_colors)

    ani = animation.FuncAnimation(fig, update, fargs=(player_data, ball_data, los, togo_line, play_desc, team_colors, ax),
                                  frames=range(int(player_data['frame'].min()), int(player_data['frame'].max()) + 1),
                                  interval=100, repeat=False)

    ani.save('./Visualization/Images/game_animation.gif', writer='imagemagick', fps=10)  # Save as GIF

    plt.show()
