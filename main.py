# -*- coding: utf-8 -*-

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from typing import List, Dict

# Initialize the API
app = FastAPI()

# Global variables to store the model and data
data = None
user_profiles = None
target_features = ['match_sportId', 'match_categoryId', 'match_tournamentId', 'oddField_oddTypeId']

# Route to upload a new dataset and rerun the model
@app.post("/upload_json/")
async def upload_json(file: UploadFile = File(...)):
    if file.content_type != 'application/json':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JSON file.")

    contents = await file.read()
    new_json_data = json.loads(contents)

    try:
        
        if os.path.exists('uploaded_data.json'): # if uploaded_data.json is found
            with open('uploaded_data.json', 'r') as f: # open it
                existing_json_data = json.load(f)   
            data = merge_json_data(existing_json_data, new_json_data, 'uploaded_data.json')
        else:
            # Save new JSON data to the file if it doesn't exist
            with open('uploaded_data.json', 'w') as json_file:
                json.dump(new_json_data, json_file, ensure_ascii=False, indent=2)
            data = new_json_data    
            
        # Generate user profiles
        clustering_data = preprocess_data_kmeans(data)
        user_profiles = clustering(clustering_data)
        user_profiles_json = user_profiles.to_json(orient='records', indent=2)
        with open('user_profiles.json', 'w') as json_file:
            json_file.write(user_profiles_json)

        return {"message": "JSON file processed successfully", "data_preview": data[:5]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing JSON file: {e}")

def merge_bets(existing_bets: List[Dict], new_bets: List[Dict]) -> List[Dict]:
    # Create a set of unique bets
    unique_bets = {json.dumps(bet, sort_keys=True): bet for bet in existing_bets}
    for bet in new_bets:
        bet_key = json.dumps(bet, sort_keys=True)
        if bet_key not in unique_bets:
            unique_bets[bet_key] = bet
    return list(unique_bets.values())

def merge_json_data(existing_data: List[Dict], new_data: List[Dict], output_json_path: str) -> List[Dict]:
    # Combine existing and new data
    combined_data = existing_data + new_data

    # Ensure unique entries by using a set of unique keys, allowing for missing 'match_id'
    unique_entries = {}

    for entry in combined_data:
        key = (entry['userid'], entry['time'])
        if 'match_id' in entry:
            key += (entry['match_id'],)
        
        # Preserve the nested structure of each entry
        if key not in unique_entries:
            unique_entries[key] = entry
        else:
            # Merge bets if the entry already exists
            existing_bets = unique_entries[key].get('bet', [])
            new_bets = entry.get('bet', [])
            unique_entries[key]['bet'] = merge_bets(existing_bets, new_bets)

    # Convert back to a list
    merged_data = list(unique_entries.values())

    # Save the merged data to JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(merged_data, json_file, ensure_ascii=False, indent=2)

    return merged_data




# Route to get all recommendations
@app.get("/recommend/all")
async def recommend_all():
    try:
        with open('uploaded_data.json', 'r') as f:
            json_data = json.load(f)
        
        data = preprocess_data_kmeans(json_data)

        with open('user_profiles.json', 'r') as f:
            json_data = json.load(f)
        
        user_profiles = pd.json_normalize(json_data)
        
        all_users_recommendations = recommend_bets_for_all_users(user_profiles, data, target_features)
    
        return JSONResponse(content=json.loads(all_users_recommendations))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations for all users: {e}")

# Route to get single recommendations
@app.get("/recommend/{user_id}")
async def recommend(user_id: int):
    try:
        with open('uploaded_data.json', 'r') as f:
            json_data = json.load(f)
        
        data = preprocess_data_kmeans(json_data)

        with open('user_profiles.json', 'r') as f:
            json_data = json.load(f)
        
        user_profiles = pd.json_normalize(json_data)

        recommended_bets = recommend_bets(user_id, user_profiles, data, target_features)
        return JSONResponse(content=[recommended_bets])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {e}")

# Data preprocessing
def preprocess_data(json_data: List[Dict]) -> List[Dict]:
    # Flatten the JSON structure if necessary
    flattened_data = []
    for entry in json_data:
        base = {'time': entry['time'], 'userid': entry['userid']}
        for bet in entry.get('bet', []):
            bet_base = base.copy()
            bet_base['stake'] = bet.get('stake')
            for pick in bet.get('pick', []):
                pick_base = bet_base.copy()
                match = pick.get('match', {})
                market = pick.get('market', {})
                odd_field = pick.get('oddField', {})
                
                pick_base.update({
                    'match_id': match.get('id'),
                    'match_dateofmatch': match.get('dateofmatch'),
                    'match_home': match.get('home'),
                    'match_homeId': match.get('homeId'),
                    'match_away': match.get('away'),
                    'match_awayId': match.get('awayId'),
                    'match_sport': match.get('sport'),
                    'match_category': match.get('category'),
                    'match_tournament': match.get('tournament'),
                    'match_sportId': match.get('sportId'),
                    'match_categoryId': match.get('categoryId'),
                    'match_tournamentId': match.get('tournamentId'),
                    'market_freetext': market.get('freetext'),
                    'market_specialoddsvalue': market.get('specialoddsvalue'),
                    'market_typeid': market.get('typeid'),
                    'oddField_oddTypeId': odd_field.get('oddTypeId'),
                    'oddField_type': odd_field.get('type'),
                    'oddField_value': odd_field.get('value')
                })
                flattened_data.append(pick_base)
    return flattened_data

def preprocess_data_kmeans(json_data):
    
    df = pd.json_normalize(json_data, 'bet', ['time', 'userid'])
#
    df['match_id'] = df['pick'].apply(lambda x: x[0]['match']['id'])
    df['match_dateofmatch'] = df['pick'].apply(lambda x: x[0]['match']['dateofmatch'])
    df['match_home'] = df['pick'].apply(lambda x: x[0]['match']['home'])
    df['match_homeId'] = df['pick'].apply(lambda x: x[0]['match']['homeId'])
    df['match_away'] = df['pick'].apply(lambda x: x[0]['match']['away'])
    df['match_awayId'] = df['pick'].apply(lambda x: x[0]['match']['awayId'])
    df['match_sport'] = df['pick'].apply(lambda x: x[0]['match']['sport'])
    df['match_category'] = df['pick'].apply(lambda x: x[0]['match']['category'])
    df['match_tournament'] = df['pick'].apply(lambda x: x[0]['match']['tournament'])
    df['match_sportId'] = df['pick'].apply(lambda x: x[0]['match']['sportId'])
    df['match_categoryId'] = df['pick'].apply(lambda x: x[0]['match']['categoryId'])
    df['match_tournamentId'] = df['pick'].apply(lambda x: x[0]['match']['tournamentId'])
    df['market_freetext'] = df['pick'].apply(lambda x: x[0]['market']['freetext'])
    df['market_specialoddsvalue'] = df['pick'].apply(lambda x: x[0]['market']['specialoddsvalue'])
    df['market_typeid'] = df['pick'].apply(lambda x: x[0]['market']['typeid'])
    df['oddField_oddTypeId'] = df['pick'].apply(lambda x: x[0]['oddField']['oddTypeId'])
    df['oddField_type'] = df['pick'].apply(lambda x: x[0]['oddField']['type'])
    df['oddField_value'] = df['pick'].apply(lambda x: x[0]['oddField']['value'])

    # Drop the 'pick' column
    df.drop(['pick'], axis=1, inplace=True)
    df['time'] = pd.to_datetime(df['time'])
    df['match_dateofmatch'] = pd.to_datetime(df['match_dateofmatch'])
    return df

# Clustering with Kmeans
def clustering(df):
    user_profiles = df.groupby('userid').agg({
        'stake': ['mean', 'sum', 'count'],
        'match_sport': lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown',
        'match_tournament': lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown',
        'oddField_type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown',
        'match_category': lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'
    }).reset_index()
    user_profiles.columns = ['userid', 'avg_stake', 'total_stake', 'bet_count', 'favorite_sport', 'favorite_tournament', 'favorite_oddField_type', 'favorite_match_category']
    numerical_features = ['avg_stake', 'total_stake', 'bet_count']
    categorical_features = ['favorite_sport', 'favorite_tournament', 'favorite_oddField_type', 'favorite_match_category']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    user_profiles_transformed = pipeline.fit_transform(user_profiles)
    kmeans = KMeans(n_clusters=3, random_state=42)
    user_profiles['cluster'] = kmeans.fit_predict(user_profiles_transformed)
    return user_profiles


# Single user recommendations
def recommend_bets(user_id, user_profiles, df, target_features):
    
    # df = df.groupby('userid').filter(lambda x: len(x) > 2)
    user_cluster = user_profiles[user_profiles['userid'] == user_id]['cluster'].values[0]
    similar_users = user_profiles[user_profiles['cluster'] == user_cluster]['userid'].values
    user_bets = df[df['userid'] == user_id]['match_id'].unique()
    similar_users_bets = df[df['userid'].isin(similar_users) & ~df['match_id'].isin(user_bets)]
    recommendations = {}
    for feature in target_features:
        recommended_bets = similar_users_bets[feature].value_counts().head(3).index.tolist()
        recommendations[feature] = recommended_bets
    return recommendations

# Recommendations for all users
def recommend_bets_for_all_users(user_profiles, df, target_features):
    all_users_results = []
    user_ids = user_profiles['userid'].unique()
    
    for user_id in user_ids:
        try:
            user_cluster = user_profiles[user_profiles['userid'] == user_id]['cluster'].values[0]
            similar_users = user_profiles[user_profiles['cluster'] == user_cluster]['userid'].values
            user_bets = df[df['userid'] == user_id]['match_id'].unique()
            similar_users_bets = df[df['userid'].isin(similar_users) & ~df['match_id'].isin(user_bets)]
            
            recommendations = {'userid': int(user_id)}  # Ensure user_id is a native int
            for feature in target_features:
                recommended_bets = similar_users_bets[feature].value_counts().head(3).index.tolist()
                recommendations[feature] = [int(x) if isinstance(x, np.integer) else x for x in recommended_bets]
            
            all_users_results.append(recommendations)
        except Exception as e:
            print(f"Error processing user_id {user_id}: {e}")
    
    all_users_results_json = json.dumps(all_users_results, ensure_ascii=False)
    return all_users_results_json


    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
