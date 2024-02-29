import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

def animate_tracking_data(play_data):
    # Get play details
    play_desc = play_data['playDescription'].iloc[0]
    play_dir = play_data['playDirection'].iloc[0]
    yards_togo = play_data['yardsToGo'].iloc[0]
    los = play_data['absoluteYardlineNumber'].iloc[0]
    togo_line = los-yards_togo if play_dir=="left" else los+yards_togo

    # Separate player and ball tracking data
    player_data = play_data[play_data['displayName'] != "ball"]
    ball_data = play_data[play_data['displayName'] == "ball"]

    # Compute velocity components
    player_data.loc[:, 'dir_rad'] = np.deg2rad(player_data['dir'])
    player_data.loc[:, 'v_x'] = np.sin(player_data['dir_rad']) * player_data['s']
    player_data.loc[:, 'v_y'] = np.cos(player_data['dir_rad']) * player_data['s']

    player_data.loc[:, 'isFastestFlag'] = player_data.groupby(['frame', 'teamAbbr'])['s'].transform('max') == player_data['s']


    # Load the image
    img = mpimg.imread('./Scripts/Results/Visualization/court.png')

    # Animation
    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        ax.imshow(img)  # Display the image in each frame
        df = player_data[player_data['frame'] == i]
        ax.scatter(df['x'], df['y'], c=df['isFastestFlag'])
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)

    animation.FuncAnimation(fig, animate, frames=range(int(player_data['frame'].min()), int(player_data['frame'].max())+1), interval=100)

    plt.show()