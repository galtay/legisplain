from embedding_atlas.projection import compute_text_projection
import pandas as pd

path = "/Users/galtay/repos/legisplain/congress-hf/usc-vecs-s8192-o512-google-embeddinggemma-300m/data"
df = pd.read_parquet(path)
df = compute_text_projection(df, text="text",
    x="projection_x", y="projection_y", neighbors="neighbors"
)