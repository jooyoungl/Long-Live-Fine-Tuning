import pandas as pd
from pathlib import Path
import praw
import datetime
import json
import networkx as nx
import matplotlib.pyplot as plt
import time

""" Import configuration for dataset and output classification """
# from configs.ws_config import DATA_CSV, OUTPUT_JSON # WATER SCANDAL
# from configs.tb_config import DATA_CSV, OUTPUT_JSON # TRUMP BLEACH
from configs.it_config import DATA_CSV, OUTPUT_JSON # IMMIGATION TATTOOS


# Reddit API credentials
CLIENT_ID = 'sWmzMAhmTGSN7y5NMMto2Q'
CLIENT_SECRET = 'rNF04vfQTLDzb_5T2LHBZABXt35iyA'
USERNAME = "Fluffy_Win4717"
PASSWORD = "wCu_EXEp.Cj2_6x"
USER_AGENT = "windows:reddit_dt_sns_analysis:v0.1 (by u/Fluffy_Win4717)"

# Setup Reddit API
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    password=PASSWORD,
    user_agent=USER_AGENT,
    username=USERNAME,
    ratelimit_seconds=300,
)

# Load Reddit post IDs
reddit_df = pd.read_csv(DATA_CSV)


# Iterate over posts and build reply networks
for i, post_id in enumerate(reddit_df['id']):
    json_path = OUTPUT_JSON / f"{post_id}.json"
    if json_path.exists():
        # Skip posts that have already been processed
        continue  

    start = time.perf_counter()

    # Load submission and fetch all comments
    submission = reddit.submission(post_id)  
    submission.comments.replace_more(limit=None)
    
    #  Prepare data structures
    author_lookup = {}
    edges = []
    comments_data = []
    
    # Breadth-first traversal of all comments
    comment_queue = submission.comments[:]
    while comment_queue:
        comment = comment_queue.pop(0)
        author = str(comment.author)
        parent_id = comment.parent_id.split("_")[-1]  # Get parent comment ID

        # Map comment ID to author
        author_lookup[comment.id] = author

        # Store comment details
        comments_data.append({
            "id": comment.id,
            "author": author,
            "body": comment.body,
            "score": comment.score,
            "parent_id": parent_id,
            "depth": comment.depth,
            "created_utc": datetime.datetime.fromtimestamp(
                comment.created_utc, tz=datetime.timezone.utc
            ).isoformat(),
        })

        # Add edge in network if the parent is another comment
        if comment.parent_id.startswith("t1_") and author != "None":
            parent_author = author_lookup.get(parent_id)
            if parent_author and parent_author != "None":
                edges.append({
                    "source": author,
                    "target": parent_author,
                    "comment_id": comment.id
                })

        # Add replies to queue
        comment_queue.extend(comment.replies)

    # Build reply network graph
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge["source"], edge["target"], comment_id=edge["comment_id"])

    # Save JSON output
    output = {
        "submission": {
            "id": submission.id,
            "title": submission.title,
            "url": submission.url,
            "score": submission.score,
            "num_comments": submission.num_comments,
            "author": str(submission.author),
            "created_utc": datetime.datetime.fromtimestamp(
                submission.created_utc, tz=datetime.timezone.utc
            ).isoformat(),
        },
        "comments": comments_data,
        "reply_network": edges
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - start
    print(f"✅ {post_id} saved ({elapsed:.2f}s) | Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # Save PNG visualization of reply network
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="lightblue", arrowsize=20)
    plt.title("Reddit Reply Network")
    plt.savefig(OUTPUT_JSON / f"{post_id}.png")
    plt.close()

    # Update existing CSV columns
    reddit_df.at[i, "nodes"] = G.number_of_nodes()
    reddit_df.at[i, "edges"] = G.number_of_edges()
    reddit_df.at[i, "time"] = round(elapsed, 3)

# Save CSV with updated values
reddit_df.to_csv(DATA_CSV, index=False)
print("✅ CSV updated with network statistics.")