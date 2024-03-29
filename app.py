from bs4 import BeautifulSoup
import requests
import streamlit as st
import re
import pandas as pd
import numpy as np
import statsmodels.api as sm
# nba api
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_curve,auc,recall_score,f1_score,precision_score,classification_report,confusion_matrix,auc
from time import sleep

st.set_page_config(
    page_title='Sportnalytics',
    layout="centered"
    )
class NBAPredictor:
    def __init__(self):
        # create dictionary of nba teams
        self.nba_teams = {}
        self.df = None
        
    def extract_api_data(self,debugging=True):
        nbateams = pd.DataFrame(teams.get_teams())
        team_abrs = nbateams['abbreviation'].unique()
        
        for team_abr in team_abrs:  # populate nba_teams dict
            self.nba_teams[team_abr] = {'GAME_PLAYED': 0, 
                                'BEF_AVG_PTS':0,
                                'BEF_AVG_AST':0,
                                'BEF_AVG_OREB':0,
                                'BEF_AVG_DREB':0,
                                'AFT_AVG_PTS':0,
                                'AFT_AVG_AST':0,
                                'AFT_AVG_OREB':0,
                                'AFT_AVG_DREB':0,
                                'cPTS':0,
                                'cAST':0,
                                'cOREB':0,
                                'cDREB':0,
                                "cFGA":0,
                                "cTO": 0,
                                "cFTA":0,
                                'OFFRATE':0,
                                'DEFRATE':0,
                                "ELO": 1500,
                                'DIS_PTS' : 0,
                                'DIS_AST' : 0,
                                'DIS_OREB' : 0,
                                'DIS_DREB' : 0,}
        
        team_ids = nbateams['id'].unique()
        df = pd.DataFrame()
        
        if debugging == True:
            df = pd.read_csv('nba_2020.csv')
        else:
            for team_id in team_ids:
                gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
                games = gamefinder.get_data_frames()[0]
                games = games[games['GAME_DATE'] > '2020-12-22'] # NBA 20/21 season starts from this date
                df = df.append(games)
                sleep(1)

        # df = df.sort_values('GAME_DATE',ascending=True)
        self.df = df
        return 

    def clean_data(self):
        master_data = self.df.copy()
        master_data['GAME_DATE'] = pd.to_datetime(master_data['GAME_DATE']) # change GAME_DATE to datetime type
        master_data = master_data.sort_values(by = "GAME_DATE", ascending = True)
        master_data['SEASON_ID'] = master_data['SEASON_ID'].astype(str) # change SEASON_ID to str type
        
        # adding columns to the dataframe
        master_data['HOME_COURT'] = np.where(master_data['MATCHUP'].str.contains('vs.'), 1, 0)      # home team = 1, away team = 0
        
        for variable in self.nba_teams['ATL']:
            master_data[variable] = 0
            
        master_data['DIS_PTS'] = 0
        master_data['DIS_AST'] = 0
        master_data['DIS_OREB'] = 0
        master_data['DIS_DREB'] = 0
        master_data['DIS_OFFRATE'] = 0
        master_data['DIS_DEFRATE'] = 0
        master_data['DIS_ELO'] = 0
        
        # merging rows of the same game into 1 row
        master_data_combined = master_data.merge(master_data, on='GAME_ID')
        master_data_combined = master_data_combined.drop(master_data_combined[master_data_combined['TEAM_ID_x'] == master_data_combined['TEAM_ID_y']].index)
        master_data_combined = master_data_combined.iloc[::2]

        master_data_combined = master_data_combined.replace(['W','L'], [int(1), int(0)]) # win = 1, lose = 0
        
        final_df = self.update_dataframe(master_data_combined) # update values for all games
        
        final_df.drop(final_df[(final_df['GAME_PLAYED_x'] == 1) | (final_df['GAME_PLAYED_y'] == 1 )].index, inplace=True) # omit first games of all teams
        
        final_df.dropna(inplace = True) # drop games that don't have WL value
        
        self.final_df = final_df
        return 

    def update_dataframe(self,df):
        nba_teams = self.nba_teams
        for i, row in df.iterrows():
        #   get the name of both teams
            team_1 = row['TEAM_ABBREVIATION_x']
            team_2 = row['TEAM_ABBREVIATION_y']

        #   Team 1
        #   increase count of games played 
            nba_teams[team_1]['GAME_PLAYED'] += 1
            df.loc[i,'GAME_PLAYED_x'] = nba_teams[team_1]['GAME_PLAYED']

        #   add a before avg ratings based on previous games
            nba_teams[team_1]['BEF_AVG_PTS'] = nba_teams[team_1]['AFT_AVG_PTS']
            nba_teams[team_1]['BEF_AVG_AST'] = nba_teams[team_1]['AFT_AVG_AST']
            nba_teams[team_1]['BEF_AVG_OREB'] = nba_teams[team_1]['AFT_AVG_OREB']
            nba_teams[team_1]['BEF_AVG_DREB'] = nba_teams[team_1]['AFT_AVG_DREB']

            nba_teams[team_1]['cPTS'] += row['PTS_x']
            nba_teams[team_1]['cAST'] += row['AST_x']
            nba_teams[team_1]['cOREB'] += row['OREB_x']
            nba_teams[team_1]['cDREB'] += row['DREB_x']
            nba_teams[team_1]['cFGA'] += row['FGA_x']
            nba_teams[team_1]['cTO'] += row['TOV_x']
            nba_teams[team_1]['cFTA'] += row['FTA_x']

            nba_teams[team_1]['AFT_AVG_PTS'] = nba_teams[team_1]['cPTS'] /nba_teams[team_1]["GAME_PLAYED"]
            nba_teams[team_1]['AFT_AVG_AST'] = nba_teams[team_1]['cAST']/nba_teams[team_1]["GAME_PLAYED"]
            nba_teams[team_1]['AFT_AVG_OREB'] = nba_teams[team_1]['cOREB']/nba_teams[team_1]["GAME_PLAYED"]
            nba_teams[team_1]['AFT_AVG_DREB'] = nba_teams[team_1]['cDREB']/nba_teams[team_1]["GAME_PLAYED"]
            
        #       calculating disparity between both teams for team 1
            nba_teams[team_1]['DIS_PTS'] = nba_teams[team_1]['BEF_AVG_PTS'] - nba_teams[team_2]['BEF_AVG_PTS']
            nba_teams[team_1]['DIS_AST'] = nba_teams[team_1]['BEF_AVG_AST'] - nba_teams[team_2]['BEF_AVG_AST']
            nba_teams[team_1]['DIS_OREB'] = nba_teams[team_1]['BEF_AVG_OREB'] - nba_teams[team_2]['BEF_AVG_OREB']
            nba_teams[team_1]['DIS_DREB'] = nba_teams[team_1]['BEF_AVG_DREB'] - nba_teams[team_2]['BEF_AVG_DREB']

        #   Team 2
        #       increase count of games played 
            nba_teams[team_2]['GAME_PLAYED'] += 1
            df.loc[i,'GAME_PLAYED_y'] = nba_teams[team_2]['GAME_PLAYED']   

        #       add a before avg ratings for the previous gams 
            nba_teams[team_2]['BEF_AVG_PTS'] = nba_teams[team_2]['AFT_AVG_PTS']
            nba_teams[team_2]['BEF_AVG_AST'] = nba_teams[team_2]['AFT_AVG_AST']
            nba_teams[team_2]['BEF_AVG_OREB'] = nba_teams[team_2]['AFT_AVG_OREB']
            nba_teams[team_2]['BEF_AVG_DREB'] = nba_teams[team_2]['AFT_AVG_DREB']

            nba_teams[team_2]['cPTS'] += row['PTS_y']
            nba_teams[team_2]['cAST'] += row['AST_y']
            nba_teams[team_2]['cOREB'] += row['OREB_y']
            nba_teams[team_2]['cDREB'] += row['DREB_y']
            nba_teams[team_2]['cFGA'] += row['FGA_y']
            nba_teams[team_2]['cTO'] += row['TOV_y']
            nba_teams[team_2]['cFTA'] += row['FTA_y']

            nba_teams[team_2]['AFT_AVG_PTS'] = nba_teams[team_2]['cPTS'] /nba_teams[team_2]["GAME_PLAYED"]
            nba_teams[team_2]['AFT_AVG_AST'] = nba_teams[team_2]['cAST']/nba_teams[team_2]["GAME_PLAYED"]
            nba_teams[team_2]['AFT_AVG_OREB'] = nba_teams[team_2]['cOREB']/nba_teams[team_2]["GAME_PLAYED"]
            nba_teams[team_2]['AFT_AVG_DREB'] = nba_teams[team_2]['cDREB']/nba_teams[team_2]["GAME_PLAYED"]
            
        #       calculating disparity between both teams for team 2
            nba_teams[team_2]['DIS_PTS'] = nba_teams[team_2]['BEF_AVG_PTS'] - nba_teams[team_1]['BEF_AVG_PTS']
            nba_teams[team_2]['DIS_AST'] = nba_teams[team_2]['BEF_AVG_AST'] - nba_teams[team_1]['BEF_AVG_AST']
            nba_teams[team_2]['DIS_OREB'] = nba_teams[team_2]['BEF_AVG_OREB'] - nba_teams[team_1]['BEF_AVG_OREB']
            nba_teams[team_2]['DIS_DREB'] = nba_teams[team_2]['BEF_AVG_DREB'] - nba_teams[team_1]['BEF_AVG_DREB']

        #   Both Teams    
        #       calculating pre-game off def ratings of both teams
            df.loc[i,"OFFRATE_x"] = nba_teams[team_1]['OFFRATE']
            df.loc[i,'DEFRATE_x'] = nba_teams[team_1]['DEFRATE']
            df.loc[i,"OFFRATE_y"] = nba_teams[team_2]['OFFRATE']
            df.loc[i,'DEFRATE_y'] = nba_teams[team_2]['DEFRATE']       
            df.loc[i,'DIS_OFFRATE_x'] = nba_teams[team_1]['OFFRATE'] - nba_teams[team_2]['OFFRATE']
            df.loc[i,'DIS_DEFRATE_x'] = nba_teams[team_1]['DEFRATE'] - nba_teams[team_2]['DEFRATE']    
            df.loc[i,'DIS_OFFRATE_y'] = nba_teams[team_2]['OFFRATE'] - nba_teams[team_1]['OFFRATE']
            df.loc[i,'DIS_DEFRATE_y'] = nba_teams[team_2]['DEFRATE'] - nba_teams[team_1]['DEFRATE']
        
        #       updating post-game off def ratings of both teams in the dictionary
            tot_pos_1 = nba_teams[team_1]['cFGA'] - nba_teams[team_1]['cOREB'] + nba_teams[team_1]['cTO'] +(0.4* nba_teams[team_1]['cFTA'])
            off_ratings_1 = nba_teams[team_1]['cPTS']/tot_pos_1
            nba_teams[team_1]['OFFRATE'] = off_ratings_1
            def_ratings_1 = nba_teams[team_2]['cPTS']/tot_pos_1
            nba_teams[team_1]['DEFRATE'] = def_ratings_1

            tot_pos_2 = nba_teams[team_2]['cFGA'] - nba_teams[team_2]['cOREB'] + nba_teams[team_2]['cTO'] +(0.4* nba_teams[team_2]['cFTA'])
            off_ratings_2 = nba_teams[team_2]['cPTS']/tot_pos_2
            nba_teams[team_2]['OFFRATE'] = off_ratings_2
            def_ratings_2 = nba_teams[team_1]['cPTS']/tot_pos_2
            nba_teams[team_2]['DEFRATE'] = def_ratings_2

        #       calculating pre-game elo of both teams
            df.loc[i, 'ELO_x'] = nba_teams[team_1]['ELO']
            df.loc[i, 'ELO_y'] = nba_teams[team_2]['ELO']
            df.loc[i,'DIS_ELO_x'] = nba_teams[team_1]['ELO'] - nba_teams[team_2]['ELO']
            df.loc[i,'DIS_ELO_y'] = nba_teams[team_2]['ELO'] - nba_teams[team_1]['ELO']                                        

            K_FACTOR = 20       # constant value for multiplier

            P_team = 1/(1 + 10 ** ((nba_teams[team_2]['ELO'] - nba_teams[team_1]['ELO'])/400))      # probability of team winning

            if row['WL_x'] == 1:
                elo_change = K_FACTOR * (1 - P_team)        # formula for change in elo if team 1 wins
            else:
                elo_change = K_FACTOR * (0 - P_team)        # formula for change in elo if team 1 loses

        #       updating post-game elo of both teams in the dictionary
            nba_teams[team_1]['ELO'] += elo_change
            nba_teams[team_2]['ELO'] -= elo_change


        #       update the values for each row
            df.loc[i,'BEF_AVG_PTS_x'] = nba_teams[team_1]['BEF_AVG_PTS']
            df.loc[i,'BEF_AVG_PTS_y'] = nba_teams[team_2]['BEF_AVG_PTS']
            df.loc[i,'BEF_AVG_AST_x'] = nba_teams[team_1]['BEF_AVG_AST']
            df.loc[i,'BEF_AVG_AST_y'] = nba_teams[team_2]['BEF_AVG_AST']
            df.loc[i,'BEF_AVG_OREB_x'] = nba_teams[team_1]['BEF_AVG_OREB']
            df.loc[i,'BEF_AVG_OREB_y'] = nba_teams[team_2]['BEF_AVG_OREB']
            df.loc[i,'BEF_AVG_DREB_x'] = nba_teams[team_1]['BEF_AVG_DREB']
            df.loc[i,'BEF_AVG_DREB_y'] = nba_teams[team_2]['BEF_AVG_DREB']

            df.loc[i,'AFT_AVG_PTS_x'] = nba_teams[team_1]['AFT_AVG_PTS']
            df.loc[i,'AFT_AVG_PTS_y'] = nba_teams[team_2]['AFT_AVG_PTS']
            df.loc[i,'AFT_AVG_AST_x'] = nba_teams[team_1]['AFT_AVG_AST']
            df.loc[i,'AFT_AVG_AST_y'] = nba_teams[team_2]['AFT_AVG_AST']
            df.loc[i,'AFT_AVG_OREB_x'] = nba_teams[team_1]['AFT_AVG_OREB']
            df.loc[i,'AFT_AVG_OREB_y'] = nba_teams[team_2]['AFT_AVG_OREB']
            df.loc[i,'AFT_AVG_DREB_x'] = nba_teams[team_1]['AFT_AVG_DREB']
            df.loc[i,'AFT_AVG_DREB_y'] = nba_teams[team_2]['AFT_AVG_DREB']

            df.loc[i,'cPTS_x'] = nba_teams[team_1]['cPTS']
            df.loc[i,'cAST_x'] = nba_teams[team_1]['cAST']
            df.loc[i,'cOREB_x'] = nba_teams[team_1]['cOREB']
            df.loc[i,'cDREB_x'] = nba_teams[team_1]['cDREB']
            df.loc[i,'cFGA_x'] = nba_teams[team_1]['cFGA']
            df.loc[i,'cTO_x'] = nba_teams[team_1]['cTO']
            df.loc[i,'cFTA_x'] = nba_teams[team_1]['cFTA']
            df.loc[i,'cPTS_y'] = nba_teams[team_2]['cPTS']
            df.loc[i,'cAST_y'] = nba_teams[team_2]['cAST']
            df.loc[i,'cOREB_y'] = nba_teams[team_2]['cOREB']
            df.loc[i,'cDREB_y'] = nba_teams[team_2]['cDREB']
            df.loc[i,'cFGA_y'] = nba_teams[team_2]['cFGA']
            df.loc[i,'cTO_y'] = nba_teams[team_2]['cTO']
            df.loc[i,'cFTA_y'] = nba_teams[team_2]['cFTA']

            df.loc[i,'DIS_PTS_x'] = nba_teams[team_1]['DIS_PTS']
            df.loc[i,'DIS_AST_x'] = nba_teams[team_1]['DIS_AST']
            df.loc[i,'DIS_OREB_x'] = nba_teams[team_1]['DIS_OREB']
            df.loc[i,'DIS_DREB_x'] = nba_teams[team_1]['DIS_DREB']

            df.loc[i,'DIS_PTS_y'] = nba_teams[team_2]['DIS_PTS']
            df.loc[i,'DIS_AST_y'] = nba_teams[team_2]['DIS_AST']
            df.loc[i,'DIS_OREB_y'] = nba_teams[team_2]['DIS_OREB']
            df.loc[i,'DIS_DREB_y'] = nba_teams[team_2]['DIS_DREB'] 
        return df
        
    def get_significant_variables(self):
        df = self.final_df.copy()
        
        features_list = ['DIS_ELO_x', 'HOME_COURT_x', 'DIS_OFFRATE_x', 'DIS_DEFRATE_x', 'DIS_PTS_x', 'DIS_AST_x', 'DIS_OREB_x', 'DIS_DREB_x']
        target = 'WL_x'
        
        # Creating our independent and dependent variables
        x = df[features_list]
        y = df['PLUS_MINUS_x']
        
        model = sm.OLS(y,x)
        results = model.fit()

        features_list = []
        for i in range(len(x.keys())):
            if results.pvalues[i] <= 0.1:
                features_list.append(model.exog_names[i])
        
        self.prediction_df = df 
        self.features_list = features_list
        return 
    
    def predict_result(self, home_team, away_team):
        df = self.prediction_df.copy()
        features_list = self.features_list
        matchup_data = self.get_matchup_data(home_team, away_team)

        models_dict = {
            'Linear Regression': LinearRegression(),
            'Logistic Regression':LogisticRegression(),
            'Naive Bayes':GaussianNB(),
            # 'Decision Trees':DecisionTreeClassifier(),
            'SVM linear': svm.SVC(kernel='linear'),
            'SVM rbf': svm.SVC(kernel='rbf'),
            # 'Random Forest': RandomForestClassifier(n_estimators = 100),
            # 'XGBoost': xgb.XGBClassifier(use_label_encoder=False)
        }
        
        prediction_data = {} # store prediction for each model 
        
        # game = df.iloc[-1]
        
        for model_name in models_dict:
            # X_train = df[features_list].iloc[:len(df.index)-1]
            X_train = df[features_list]
            X_test = matchup_data[features_list]
            # y_train = df['WL_x'].iloc[:len(df.index)-1]
            y_train = df['WL_x']

            m = models_dict[model_name]

            if model_name == 'Linear Regression':
                # y_train = df['PLUS_MINUS_x'].iloc[:len(df.index)-1]
                y_train = df['PLUS_MINUS_x']
            m.fit(X_train, y_train)
            prediction = m.predict(X_test)

            if model_name == 'Linear Regression':
                if prediction[0] > 0:
                    prediction[0] = 1
                else:
                    prediction[0] = 0
                        
            prediction_data[model_name] = prediction[0]
            
            # print(model_name + ':', prediction[0])
        
        final_prediction = 0
        for k, v in prediction_data.items():
            final_prediction += v
        
        final_prediction = final_prediction / 5
        # print('Average outcome score:', final_prediction)
        # print('Predicted Outcome:', round(final_prediction))
        # print('Actual Outcome:', game['WL_x'])
        return final_prediction

    def get_matchup_data(self,home_team, away_team): 
        # ideally team_x and team_y should be in abbreviation i.e. 'GSW' format 
        working_df = self.prediction_df.copy()
        home_columns = ['AFT_AVG_PTS_x','AFT_AVG_AST_x','AFT_AVG_OREB_x','AFT_AVG_DREB_x','OFFRATE_x','DEFRATE_x','ELO_x','HOME_COURT_x']
        away_columns = ['AFT_AVG_PTS_y','AFT_AVG_AST_y','AFT_AVG_OREB_y','AFT_AVG_DREB_y','OFFRATE_y','DEFRATE_y','ELO_y','HOME_COURT_y']

        home_data = pd.DataFrame(working_df[working_df['TEAM_ABBREVIATION_x'] == home_team].iloc[-1,:]).transpose()
        away_data = pd.DataFrame(working_df[working_df['TEAM_ABBREVIATION_y'] == away_team].iloc[-1,:]).transpose()
        matchup_data = pd.DataFrame(home_data[home_columns].values-away_data[away_columns].values,columns=['DIS_PTS_x', 'DIS_AST_x','DIS_OREB_x', 'DIS_DREB_x', 'DIS_OFFRATE_x', 'DIS_DEFRATE_x','DIS_ELO_x','HOME_COURT_x'])
            
        return matchup_data

    def show_historical_performance(self):
        features_list = ['DIS_ELO_x', 'HOME_COURT_x', 'DIS_OFFRATE_x', 'DIS_DEFRATE_x', 'DIS_PTS_x', 'DIS_AST_x', 'DIS_OREB_x', 'DIS_DREB_x']
        target = 'WL_x'

        df = self.final_df.copy()
        # predict and test 2nd half of season
        test_data = df[(df['GAME_PLAYED_x'] > 41) & (df['GAME_PLAYED_y'] > 41 )]
        test_data['Prediction'] = 0

        models_dict = {
            'Linear Regression': LinearRegression(),
            'Logistic Regression':LogisticRegression(),
            'Naive Bayes':GaussianNB(),
            'SVM linear': svm.SVC(kernel='linear'),
            'SVM rbf': svm.SVC(kernel='rbf'),
        }

        for index, row in test_data.iterrows():
            # Creating training dataset for all previous games
            idx = df.index[df['GAME_ID'] == row['GAME_ID']]
            train_df = df.loc[:idx[0]-1]
            test_df = row.to_frame().T

            # Creating our independent and dependent variables
            x = train_df[features_list]
            y = train_df['PLUS_MINUS_x']

            model = sm.OLS(y,x)
            results = model.fit()

            # Extracting significant features
            features_list = []
            for i in range(len(x.keys())):
                if results.pvalues[i] <= 0.1:
                    features_list.append(model.exog_names[i])

            # Predicting game outcome
            prediction_data = {}
            for model_name in models_dict:
                X_train = train_df[features_list]
                y_train = train_df['WL_x']
                X_test = test_df[features_list]

                m = models_dict[model_name]

                if model_name == 'Linear Regression':
                    y_train = train_df['PLUS_MINUS_x']
                
                m.fit(X_train, y_train)
                prediction = m.predict(X_test)

                if model_name == 'Linear Regression':
                    if prediction[0] > 0:
                        prediction[0] = 1
                    else:
                        prediction[0] = 0
                                
                prediction_data[model_name] = prediction[0]
                
            final_prediction = 0
            for k, v in prediction_data.items():
                final_prediction += v

            final_prediction = round(final_prediction / 5)

            # Appending predicted outcome to test_data
            test_data['Prediction'][index] = final_prediction

        y_test = test_data['WL_x']
        f1 = f1_score(y_test,test_data['Prediction'])
        accuracy = accuracy_score(y_test,test_data['Prediction'])
        precision = precision_score(y_test,test_data['Prediction'])
        recall = recall_score(y_test,test_data['Prediction'])
        test_data = test_data[['TEAM_NAME_x','TEAM_NAME_y','GAME_DATE_x','Prediction','WL_x']]
        test_data['Correct(T/F)'] = np.where(test_data['Prediction']==test_data['WL_x'],"True","False")
        test_data.columns = ['Home','Away','Game Date','Prediction','Actual','Correct(T/F)']
        test_data.reset_index(inplace=True,drop=True)
        return test_data,f1,accuracy,precision,recall,features_list

    def show_training_performance(self):
            features_list = ['DIS_ELO_x', 'HOME_COURT_x', 'DIS_OFFRATE_x', 'DIS_DEFRATE_x', 'DIS_PTS_x', 'DIS_AST_x', 'DIS_OREB_x', 'DIS_DREB_x']
            target = 'WL_x'

            df = self.final_df.copy()
            # predict and test 2nd half of season
            train_data = df
            models_dict = {
                'Linear Regression': LinearRegression(),
                'Logistic Regression':LogisticRegression(),
                'Naive Bayes':GaussianNB(),
                'SVM linear': svm.SVC(kernel='linear'),
                'SVM rbf': svm.SVC(kernel='rbf'),
            }

            # Creating our independent and dependent variables
            x = train_data[features_list]
            y = train_data['PLUS_MINUS_x']

            model = sm.OLS(y,x)
            results = model.fit()

            # Extracting significant features
            features_list = []
            for i in range(len(x.keys())):
                if results.pvalues[i] <= 0.1:
                    features_list.append(model.exog_names[i])

            # Predicting game outcome
            prediction_data = {}
            for model_name in models_dict:
                X_train = train_data[features_list]
                y_train = train_data['WL_x']
                m = models_dict[model_name]

                if model_name == 'Linear Regression':
                    y_train = train_data['PLUS_MINUS_x']
                
                m.fit(X_train, y_train)
                prediction = m.predict(X_train)

                if model_name == 'Linear Regression':
                    prediction = np.where(prediction > 0, 1,0)
                prediction_data[model_name] = prediction
                
            final_prediction = np.zeros_like(prediction)
            for k, v in prediction_data.items():
                final_prediction += v

            final_prediction = final_prediction/5
            final_prediction = np.where(final_prediction > 0, 1,0)

            y_test = train_data['WL_x']
            f1 = f1_score(y_test,final_prediction)
            accuracy = accuracy_score(y_test,final_prediction)
            return f1,accuracy

def get_matchups():
    """
    Retrieves all the matchups from ESPN website for next day games.
    The next day logic is implemented by checking if a gameday is found,
    if found, the for loop breaks.

    :return: a list containing matchup strings
    :rtype: list
    """
    matchup_list = []
    game_date = ''
    game_matchup = ''
    url = 'https://www.espn.com.sg/nba/fixtures'
    r = requests.get(url)
    soup = BeautifulSoup(r.text,parser='html.parser',features="lxml")
    game_containers = soup.findAll('table', {'class':'schedule has-team-logos align-left'})
    counter = 0
    for game in game_containers:
        try:
            if 'time' in game.thead.text:
                game_matchup = game.tbody
                game_date = soup.findAll('div', {'id':'sched-container'})[0].findAll('h2')[counter].text
        except AttributeError:
            continue
        counter += 1

        if game_date != '':
            break
    if game_matchup == '':
        game_date = 'No upcoming games.'
        return matchup_list,game_date
        
    teams_playing = game_matchup.findAll('a', {'class':'team-name'})

    # Not needed for our web app, but just filling it in here incase we need it
    time_playing = game_matchup.findAll('td', {'data-behavior':'date_time'})

    error_name = {
            "GS":"GSW",
            "SA":"SAS",
            "WSH":"WAS",
            "NO":"NOP",
            "UTAH":"UTA",
            "NY":"NYK"
        }

    for i in range(0,len(teams_playing),2):
        away = teams_playing[i].text.split()[-1]
        home = teams_playing[i+1].text.split()[-1]
        if away in error_name:
            away = error_name[away]
        if home in error_name:
            home = error_name[home]
        matchup_string = '{} (away) vs. {} (home)'.format(away, home)
        matchup_list.append(matchup_string)
    return matchup_list, game_date

def process_selected_match(selected_match):
    away = selected_match.split(' ')[0]
    home = selected_match.split(' ')[-2]
    regex = re.compile('[^a-zA-Z]')
    away = regex.sub('',away)
    home = regex.sub('',home)
    return away,home

@st.cache
def get_NBAPredictor():
    nba = NBAPredictor()
    nba.extract_api_data()
    nba.clean_data()
    nba.get_significant_variables()
    historical_df, *metrics, features_list = nba.show_historical_performance()
    # training_f1,training_acc = nba.show_training_performance()
    return nba, historical_df,metrics,features_list

matchups,game_date = get_matchups()
nba, historical_df,metrics,features_list = get_NBAPredictor()
st.title('NBA Predictor by Sportnalytics')
selected_match = st.selectbox('Select the match for {}'.format(game_date),matchups)
away,home = process_selected_match(selected_match)
final_prediction = nba.predict_result(home,away)

if final_prediction < 1:
    st.write("{} will win!".format(away))
else:
    st.write("{} will win!".format(home))

# st.subheader("Historical Performance")
# st.write("Accuracy: {:.2f}%".format(float(training_f1)*100))
# st.write("F1 Score: {:.2f}%".format(float(training_acc)*100))

st.subheader('Performance for Current Season')
st.write(historical_df)
st.write("Accuracy: {:.2f}%".format(float(metrics[1])*100))
st.write("F1 Score: {:.2f}%".format(float(metrics[0])*100))

st.title("How does it work?")
st.image("data_pipeline.png")
st.subheader("Feature Engineering Highlight - ELO")
st.write("Through our EDA, we identified that ELO was the most significant feature in determining if a team would win. Here's how we computed our ELO, drawing inspirations from [FiveThirtyEight](https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/).")
st.write("Suppose we have 2 teams, Team 1 (ELO1) and 2 (ELO2):")
st.latex("P(1) = \\frac {1} {1+10^ \\frac {ELO2-ELO1} {400}} \\text{ - Probability of Team 1 Winning.}")
st.latex("P(2) = \\frac {1} {1+10^ \\frac {ELO1-ELO2} {400}} \\text{ - Probability of Team 2 Winning.}")
st.latex("\\text{Let K} = 30")
st.latex("\\text{If Team 1 wins}: \\space ELO1 = ELO1 + K * ( 1-P(1) ) \\space ELO2 = ELO2 + K * ( 0 -P(2) )")
st.latex("\\text{If Team 2 wins}: \\space ELO1 = ELO1 + K * ( 0-P(1) ) \\space ELO2 = ELO2 + K * ( 1 -P(2) )")
st.write("At the start of the season, we set every team at 1500 ELO and dynamically compute ELO as the season progresses. This gives us a superb feature for our models to learn from.")

st.subheader("The Final Input into our Models")
st.write("Instead of using raw inputs, the team decided to compute the disparity in averages between two opposing sides. This yielded better results.")
st.write(features_list)
st.write("Surprisingly, we found that with the simple inputs of disparities in ELO, Home Court Advantage and Offensive Rating, we were able to construct a strong model that historically averages 71% in accuracy.")
st.write("Our analysis showed that additional features were adding noise to our model and thus made it less generalisable. Hence, we opted to only use these three factors, which were the most significant.")

st.title("Contributions")
st.write("This was a project as part of the Data Associate Programme by SMU BIA. In the team, we have [Brandon](https://www.linkedin.com/in/brandon-tan-jun-da/), [Jian Yu](https://www.linkedin.com/in/chen-jian-yu/), [Samuel](https://www.linkedin.com/in/samuel-sim-7368241aa/) and [Leonard](https://www.linkedin.com/in/leonard-siah-0679631a1/). Thank you for checking us out and have a nice day!")