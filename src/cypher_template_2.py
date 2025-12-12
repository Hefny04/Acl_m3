"""
Library of Cypher Query Templates for the FPL Agent.
UPDATED: Optimized for User's Schema (No Value/Ownership, Heavy on ICT/Stats).
"""

CYPHER_TEMPLATES = {
    # 1. PLAYER SUMMARY (Added Bonus, BPS, ICT)
    "player_summary": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        RETURN p.player_name AS Player,
               sum(r.total_points) AS TotalPoints,
               sum(r.goals_scored) AS Goals,
               sum(r.assists) AS Assists,
               sum(r.minutes) AS Minutes,
               sum(r.bonus) AS BonusPoints,
               sum(r.bps) AS BPS,
               sum(r.ict_index) AS TotalICT
    """,

    # 2. TOP PLAYERS BY POSITION (Standard Leaderboard)
    "top_players_by_position": """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
        WHERE toLower(pos.name) = toLower($position) OR toLower(pos.name) = toLower($position_mapped)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, pos, sum(coalesce(r.total_points, 0)) AS TotalPoints
        ORDER BY TotalPoints DESC
        LIMIT toInteger($limit)
        RETURN p.player_name AS Player, pos.name AS Position, TotalPoints
    """,

    # 3. PLAYER VS TEAM (Added Threat/Creativity to see playstyle)
    "player_vs_team": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team)
        WHERE toLower(t.name) CONTAINS toLower($opponent)
        RETURN p.player_name AS Player,
               f.fixture_number AS GW,
               t.name AS Opponent,
               r.total_points AS Points,
               r.goals_scored AS Goals,
               r.ict_index AS ICT_Index
    """,

    # 4. SQUAD LIST (Unchanged)
    "team_squad_by_position": """
        MATCH (t:Team) WHERE toLower(t.name) CONTAINS toLower($team)
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
        WHERE toLower(pos.name) = toLower($position) OR toLower(pos.name) = toLower($position_mapped)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)
        WITH p, pos, t, sum(r.total_points) as TotalPoints
        ORDER BY TotalPoints DESC
        RETURN t.name AS Team, p.player_name AS Player, pos.name AS Position, TotalPoints
        LIMIT toInteger($limit)
    """,

    # 5. COMPARE PLAYERS (Added Underlying Stats: ICT, Threat, Creativity)
    "compare_players": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE any(name IN $player_names WHERE toLower(p.player_name) CONTAINS toLower(name))
        RETURN p.player_name AS Player,
               sum(r.total_points) AS TotalPoints,
               sum(r.goals_scored) AS Goals,
               sum(r.assists) AS Assists,
               sum(r.minutes) AS Minutes,
               sum(r.ict_index) AS Total_ICT,
               sum(r.threat) AS Total_Threat,
               sum(r.creativity) AS Total_Creativity
    """,

    # 6. COMPARE RECENT FORM (Last 5 Games - Critical for decisions)
    "compare_players_last_5": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE any(name IN $player_names WHERE toLower(p.player_name) CONTAINS toLower(name))
        WITH p, f, r ORDER BY f.fixture_number DESC
        WITH p, collect(r)[0..5] as recent_games
        RETURN p.player_name AS Player,
               reduce(s=0, x in recent_games | s + x.total_points) as Points_Last_5,
               reduce(s=0, x in recent_games | s + x.goals_scored) as Goals_Last_5,
               reduce(s=0, x in recent_games | s + x.ict_index) as ICT_Last_5
    """,

    # 7. TEAM PERFORMANCE IN GW
    "team_performance_in_gw": """
        MATCH (s:Season {season_name: $season})-[:HAS_GW]->(g:Gameweek {GW_number: toInteger($gw)})
        MATCH (g)-[:HAS_FIXTURE]->(f:Fixture)
        MATCH (t:Team)<-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]-(f)
        WHERE toLower(t.name) CONTAINS toLower($team)
        MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(opponent:Team)
        WHERE opponent.name <> t.name
        MATCH (p:Player)-[r:PLAYED_IN]->(f)
        MATCH (p)-[:PLAYED_IN]->(f_all:Fixture {season: $season})
        MATCH (f_all)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)
        WITH g, t, opponent, p, r, count(f_all) as squad_games
        WHERE squad_games > 2
        WITH g, t, opponent, sum(r.goals_scored) as TeamGoals, sum(r.total_points) as TeamPoints, collect(p.player_name)[0..3] as KeyPlayers
        RETURN t.name AS Team, g.GW_number AS GW, opponent.name AS Opponent, TeamGoals, TeamPoints, KeyPlayers
    """,

    # 8. SCOUT RECOMMENDATIONS (Replaces Differentials)
    # Finds players with High ICT Index (Underlying Stats) in last 3 games
    "recommend_differentials": """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
        WHERE toLower(pos.name) = toLower($position) OR toLower(pos.name) = toLower($position_mapped)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, pos, r ORDER BY f.fixture_number DESC
        WITH p, pos, collect(r)[0..3] as last_3
        WITH p, pos, 
             reduce(s=0, x in last_3 | s + x.total_points) as form_points,
             reduce(s=0, x in last_3 | s + x.ict_index) as form_ict
        // Logic: High Underlying Stats (ICT)
        RETURN p.player_name AS Player, 
               pos.name AS Position, 
               form_points AS Points_Last_3,
               form_ict AS ICT_Last_3
        ORDER BY form_ict DESC
        LIMIT toInteger($limit)
    """,

    # 9. BEST CAPTAIN OPTIONS (Uses Form + ICT)
    "best_captain_options": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, r ORDER BY f.fixture_number DESC
        WITH p, collect(r)[0..3] as last_3_games
        WITH p, 
             reduce(s = 0, x IN last_3_games | s + x.total_points) as form_points,
             reduce(s = 0, x IN last_3_games | s + x.ict_index) as form_ict
        ORDER BY form_points + form_ict DESC
        LIMIT 5
        RETURN p.player_name AS Player, form_points AS PointsLast3, form_ict as ICTLast3
    """,

    # 10. AVAILABILITY CHECK
    "player_availability_check": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        WITH p, r, f ORDER BY f.fixture_number DESC LIMIT 3
        RETURN p.player_name AS Player, collect(r.minutes) as Last3Minutes
    """,
    
    # 11. HIGHEST SCORING GW
    "highest_scoring_gw": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        MATCH (s:Season)-[:HAS_GW]->(g:Gameweek)-[:HAS_FIXTURE]->(f)
        RETURN p.player_name AS Player, g.GW_number AS GW, r.total_points AS Points
        ORDER BY Points DESC
        LIMIT 1
    """
}