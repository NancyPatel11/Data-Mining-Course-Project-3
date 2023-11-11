# Data-Mining-Course-Project-3 - ICC Cricket World Cup 2023 Simulator

# Overview

This project is a simulation of the ICC Cricket World Cup 2023. It provides a platform for cricket enthusiasts to experience the excitement of the tournament through a computer program. The simulator uses statistical algorithms and user input to simulate matches, predict outcomes, and generate a virtual representation of the 2023 Cricket World Cup.

# Purpose

The purpose of this project is to create an engaging and realistic simulation of the ICC Cricket World Cup, allowing users to experience the thrill of the tournament in a virtual environment. It serves as a fun and educational tool for cricket fans who want to explore hypothetical scenarios, analyze match outcomes, and immerse themselves in the excitement of the game.

# Installation

To run the ICC Cricket World Cup 2023 Simulator on your machine, follow these steps:

1. *Clone the Repository:*

   in git bash write the following command:

   git clone https://github.com/NancyPatel11/Data-Mining-Course-Project-3

   cd Data-Mining-Course-Project-3

2. *Install Dependencies:*

   in git bash write the following command: 

   pip install -r requirements.txt
   
3. *Run the Simulator:*

    in git bash write the following command: 

    python api.py
   
# Prerequisites

- Python 3.6 or higher

- Git (for cloning the repository)

# External Dependencies

- Check the requirements.txt file for a list of dependencies.

# Datasets Used:

a. deliveries.csv

This dataset is taken from kaggle. It has following features:

1. `match_id`: Identifier for a specific cricket match.

2. `season`: The cricket season during which the match took place.

3. `start_date`: The date when the match started.

4. `venue`: The location or stadium where the match was played.

5. `innings`: The inning number (1st or 2nd) in the match.

6. `ball`: The ball number in the current inning.

7. `batting_team`: The team currently batting.

8. `bowling_team`: The team currently bowling.

9. `striker`: The batsman who is currently facing the ball.

10. `non_striker`: The batsman at the non-striker's end.

11. `bowler`: The bowler delivering the ball.

12. `runs_off_bat`: The runs scored off the bat (excluding extras).

13. `extras`: Extra runs scored, including wides, no-balls, byes, leg-byes, and penalties.

14. `wides`: The run for wide deliveries bowled.

15. `noballs`: The run for no-balls bowled.

16. `byes`: Runs scored due to byes (awarded when the ball passes the batsman without touching the bat).

17. `legbyes`: Runs scored due to leg-byes (awarded for runs off the batsman's body).

18. `penalty`: Penalty runs awarded to the batting team.

19. `wicket_type`: Type of wicket taken (e.g., caught, bowled, lbw).

20. `player_dismissed`: The batsman who got dismissed (if a wicket fell).

21. `other_wicket_type`: Additional information on the type of wicket, if applicable.

22. `other_player_dismissed`: Additional information on the dismissed player, if applicable.

b. points_table.csv

This dataset is taken from kaggle. It has following features:

1. `Ranking`: The position of the cricket team in the overall ranking.

2. `Team`: The name of the cricket team.

3. `Matches`: The total number of matches played by the team.

4. `Won`: The number of matches won by the team.

5. `Lost`: The number of matches lost by the team.

6. `Tie`: The number of matches ending in a tie for the team.

7. `No Results`: The number of matches with no conclusive result for the team.

8. `Points`: The total points earned by the team in the ranking system.

9. `Net Run Rate`: The team's net run rate, calculated based on runs scored and conceded.

10. `Series Form`: The team's recent performance trend in a series of matches.

11. `Next Match`: Information about the team's upcoming match.

12. `For`: The total runs scored by the team.

13. `Against`: The total runs conceded by the team.

c. matches.csv

This dataset is taken from kaggle. It has following features.

1. `season`: The cricket season in which the match is played.

2. `team1`: One of the participating teams in the match.

3. `team2`: The other participating team in the match.

4. `date`: The date on which the match is scheduled.

5. `match_number`: The unique identifier for the match in the season.

6. `venue`: The location or stadium where the match will be played.

7. `city`: The city where the match is hosted.

8. `toss_winner`: The team winning the coin toss before the match.

9. `toss_decision`: The decision made by the toss winner (batting or bowling).

10. `player_of_match`: The player adjudged as the best performer in the match.

11. `umpire1`: The name of the first on-field umpire.

12. `umpire2`: The name of the second on-field umpire.

13. `reserve_umpire`: The reserve umpire designated for the match.

14. `match_referee`: The official overseeing the match and ensuring fair play.

15. `winner`: The team that emerged victorious in the match.

16. `winner_runs`: The margin of victory for the winning team in terms of runs.

17. `winner_wickets`: The number of wickets by which the winning team secured victory.

18. `match_type`: The type of cricket match (e.g., One Day International, T20, Test).

d. upcoming_matches.csv:

We manually created this dataset for the future matches.It has the following feature:

1. `team1`: One of the participating cricket teams in a match.

2. `team2`: The other participating cricket team in a match.

3. `venue`: The location or stadium where the cricket match is being played.

e. match_summary.csv

This data is created by using match_summary_csv_creation.ipynb file. It consists of features venue, innings, batting_team, bowling_team, total_runs_per_inning_match and total_overs_played. This dataset basically shows the matchwise runs made by each team and the total overs played by each of them. This dataset is created using the matches.csv and deliveries.csv datasets taken from kaggle. The breif description of each feature follows:

1. `venue_x`: The location or stadium where the cricket match is taking place.

2. `innings`: The inning number (1 or 2) in the cricket match.

3. `batting_team`: The team currently batting in a particular inning.

4. `bowling_team`: The team currently bowling in a particular inning.

5. `total_runs_per_inning_match`: The total runs scored in a specific inning of the match by the batting team.

6. `Total_Overs_Played`: The total number of overs played in the cricket match by batting team.

f. playerwise_df.csv

This data is created using total_runs_wickets_prediction.ipynb file. It consists of features match_id, season, start_date, venue, team, opposing_team, player, total_runs and total_wickets. This dataset basically shows the matchwise runs made by the player of a team against the opposing_team. It also displays the wickets taken by the player of a team against the opposing_team. The breif description of each feature follows:

1. `match_id`: Identifier for a specific cricket match.

2. `season`: The cricket season during which the match took place.

3. `start_date`: The date when the match started.

4. `venue`: The location or stadium where the match was played.

5. `team`: The cricket team to which the player belongs.

6. `opposing_team`: The opposing cricket team in the match.

7. `player`: The name of the player.

8. `total_runs`: The total runs scored by the player in the match.

9. `total_wickets`: The total number of wickets taken by the player in the match.

# Task 1

**Problem 1:**

**Problem 2:**

# Task 2

**Finalist Team Prediction**

In order to generate the two finalist teams, we used the datasets given below:

1. upcoming_matches.csv:

2. deliveries.csv

3. points_table.csv

4. matches.csv

5. match_summary.csv

To generate the finalist teams, we used the following .ipynb files which were imported as pickle files and used in our api's 'predict' function:

1. toss_decision.ipynb

This file predicts the 'toss_decision' (field/bat) based on 'team1', 'team2', 'venue' and 'toss_winner'. 
We import this file as toss_decision.pkl and used it in api.py to predict the toss_decision for all the upcoming_matches.

2. total_run_prediction_inning_1.ipynb

This file predicts the 'total_runs_per_inning_match' (i.e the total runs made in a inning of a match) based on inputs such as 'batting_team', 'bowling_team', 'venue' and 'Total_Overs_Played'. These runs are generated only for the team batting in the first inning.
We import this file as runs_inning_1.pkl and used it in api.py to predict the runs made by team 1 in first inning for all the upcoming matches.

3. total_run_prediction_inning_2.ipynb

This file predicts the 'total_runs_per_inning_match' (i.e the total runs made in a inning of a match) based on inputs such as 'batting_team', 'bowling_team', 'venue' and 'Total_Overs_Played'. These runs are generated only for the team batting in the second inning.
We import this file as runs_inning_2.pkl and used it in api.py to predict the runs made by team 2 in second inning for all the upcoming matches.

4. total_over_pediction.ipynb
This file predicts 'Total_Overs_Played' (i.e. the number of overs faced by the batting team) based on  'batting_team', 'bowling_team', 'venue', 'innings' and 'total_runs_per_inning_match'. 
We import this file as overs.pkl and used it in api.py to predict the overs played by team 1 in inning 1 and team2 in inning 2.

Now, we checked that if the overs predicted were greater than 50, then we updated the predicted runs as predicted_runs * 50 / predicted_overs. 

Next, compare the predicted runs for both team1 and team2. The team with greated runs wins the match. 

We update the points for the winning team by adding 2. Also, we find the net run rate for each of the teams by using the formula (total runs made by the team in this match) / (total overs played by the team in this match). 

Now, we sort the points table based on points, Net Run Rate in descending order.

Next we call the 'predict_finalist' function in our api. This function takes the input as the final points table generated as above. Now, two semifinal matches will be played between:

a. team 1 and team 4 

b. team 2 and team 3

Based on their position in the points table, those teams are taken as input and the same process of predicting the runs and overs for innings 1 and 2 is done. Based on the winners of the semifinals, we get out top 2 teams as the finalists.

**Playing 11 Prediction**

In order to generate the playing eleven players for the finalist teams, we used the datasets given below:

1. playerwise_df.csv:

2. deliveries.csv

To generate the final playing eleven players for finalist teams, we used the following .ipynb files which were imported as pickle files and used in our api's 'predict_playing11' function:

1. total_runs_wickets_prediction.ipynb

This file predicts the 'total_runs' and 'total_wickets' based on 'venue', 'team', 'opposing_team' and 'player'. Basically, this file predicts the runs the player will make and the wickets the player will take in his match against the opposing team.
We import this file as runs_wickets_prediction.pkl and used it in api.py to predict the runs which will be made by each of the player of a team and also the wickets that player will take in the match. 

Based on this dataset, we now predict the final playing eleven by sorting the predicted runs and wickets in the descending order. The top eleven players with most runs and wickets will be taken as the final playing eleven for the final match.

# Task 3

**Winner Prediction**

In order to generate the winner team, we used the dataset given below:

1. matches.csv

To generate the winner team, we used the following .ipynb file which was imported as pickle files and used in our api's 'predict_winner' function:

1. toss_decision.ipynb

2. total_run_prediction_inning_1.ipynb

3. total_run_prediction_inning_2.ipynb

4. total_over_pediction.ipynb

We again use the same method to predict the winner. First we call the 'predict_finalist' function to findout the finalist teams in ICC World Cup. Now, we take these two finalist teams as input. We predict the toss decision, followed by the runs made by team 1 in inning 1 and overs played by team 1 in inning 1. Next, we find the runs made by team 2 in inning 2 and overs played by team 2 in inning 2. Based on the runs and overs, we then declare the winner.