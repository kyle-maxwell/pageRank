import pandas as pd
import numpy as np
import sys

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print('Usage: python3 pageRank.py <file>')
        sys.exit()

    d = 0.30
    epsilon = 0.0001

    filename = sys.argv[1]
    adj_matrix = process_file(filename)
    results = page_rank(adj_matrix, d, epsilon)
    print(results)

def page_rank(adj_matrix, d, epsilon):
    V = adj_matrix.shape[0]
    page_rank = pd.Series(index=adj_matrix.index.values, data=1/V)
    while True:
        print('compute iteration')
        prev_page_rank = page_rank
        page_rank = page_rank_iteration(page_rank, adj_matrix, d)
        if abs(page_rank - prev_page_rank).sum() < epsilon:
            break
    return page_rank.sort_values(ascending=False)

def page_rank_iteration(page_rank, adj_matrix, d):
    V = adj_matrix.shape[0]
    out_degree = adj_matrix.sum(axis=1)
    prestige_to_give = 1 / out_degree * page_rank
    prestige_to_recieve = adj_matrix.apply(lambda row: prestige_to_give * row, axis='rows').sum()
    return (1-d)/V + d*prestige_to_recieve

def process_file(filename):
    df = pd.read_csv(filename, header=None, engine='python',
        names=['winner', 'score1', 'loser', 'score2', 'OT'])
    df['winner'] = df['winner'].astype(str).str.replace('"', '').str.strip()
    df['loser'] = df['loser'].astype(str).str.replace('"', '').str.strip()
    unique = df['loser'].append(df['winner']).unique()
    adj_matrix = pd.crosstab(unique, unique, rownames=['losers'], colnames=['winners'])
    adj_matrix[:] = 0
    for _, game in df.iterrows():
        adj_matrix[game.winner][game.loser] = 1
    return adj_matrix


if __name__ == "__main__":
    main()