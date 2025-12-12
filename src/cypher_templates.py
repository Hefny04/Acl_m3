"""
Library of Cypher Query Templates for the FPL Agent.
UPDATED: explicit 'Subject' aliases AND 'Position' context to prevent LLM confusion.
"""

CYPHER_TEMPLATES = {
    "player_summary": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        RETURN p.player_name AS Player,
               sum(r.total_points) AS TotalPoints,
               sum(r.goals_scored) AS Goals,
               sum(r.assists) AS Assists,
               sum(r.minutes) AS Minutes
    """,
    "top_players_by_position": """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
        WHERE toLower(pos.name) = toLower($position) OR toLower(pos.name) = toLower($position_mapped)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, pos, sum(coalesce(r.total_points, 0)) AS TotalPoints
        ORDER BY TotalPoints DESC
        LIMIT toInteger($limit)
        RETURN p.player_name AS Player, pos.name AS Position, TotalPoints
    """,
    "player_vs_team": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        MATCH (f)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t:Team)
        WHERE toLower(t.name) CONTAINS toLower($opponent)
        RETURN p.player_name AS Player,
               f.fixture_number AS GW,
               t.name AS Opponent,
               r.total_points AS Points,
               r.goals_scored AS Goals
    """,
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
    "compare_players": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE any(name IN $player_names WHERE toLower(p.player_name) CONTAINS toLower(name))
        RETURN p.player_name AS Player,
               sum(r.total_points) AS TotalPoints,
               sum(r.goals_scored) AS Goals,
               sum(r.assists) AS Assists
    """,
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
    "recommend_differentials": """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
        WHERE toLower(pos.name) = toLower($position) OR toLower(pos.name) = toLower($position_mapped)
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, pos, avg(r.influence) as AvgInf, sum(r.total_points) as Points
        WHERE Points > 30 
        RETURN p.player_name AS Player, pos.name AS Position, toInteger(AvgInf) as Influence, Points
        ORDER BY AvgInf DESC
        LIMIT toInteger($limit)
    """,
    "best_captain_options": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WITH p, r ORDER BY f.fixture_number DESC
        WITH p, collect(r.total_points)[0..3] as last_3_games
        WITH p, reduce(s = 0, x IN last_3_games | s + x) as form_points
        ORDER BY form_points DESC
        LIMIT 5
        RETURN p.player_name AS Player, form_points AS FormLast3GWs
    """,
    "player_availability_check": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        WITH p, r, f ORDER BY f.fixture_number DESC LIMIT 3
        RETURN p.player_name AS Player, collect(r.minutes) as Last3Minutes
    """,
    "highest_scoring_gw": """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season: $season})
        WHERE toLower(p.player_name) CONTAINS toLower($player_name)
        MATCH (s:Season)-[:HAS_GW]->(g:Gameweek)-[:HAS_FIXTURE]->(f)
        RETURN p.player_name AS Player, g.GW_number AS GW, r.total_points AS Points
        ORDER BY Points DESC
        LIMIT 1
    """
}