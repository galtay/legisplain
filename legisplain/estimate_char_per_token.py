from pathlib import Path

import pandas as pd
import rich
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":

    congress_hf_path = Path("/Users/galtay/repos/legisplain/congress-hf")
    congress_nums = [119, 118, 117, 116, 115, 114, 113]
    nlim = None
    model_name = "BAAI/bge-small-en-v1.5"
    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer

    df = pd.DataFrame({"legis_id": [], "text": [], "num_tokens": [], "char_per_token": []})
    for congress_num in congress_nums:
        u_fpath = (
            congress_hf_path / "usc-unified" / "data" / f"usc-{congress_num}-unified.parquet"
        )
        rich.print(u_fpath)
        df_u = pd.read_parquet(u_fpath)
        df_u['text'] = df_u['tvs'].apply(lambda x: x[0]['tv_txt'])
        tokenized = tokenizer(df_u['text'].to_list())
        df_u['num_tokens'] = [len(el) for el in tokenized['input_ids']]
        df_u['char_per_token'] = df_u['text'].str.len() / df_u['num_tokens']
        df1 = df_u[['legis_id', 'text', 'num_tokens', 'char_per_token']]
        df = pd.concat([df, df1])
