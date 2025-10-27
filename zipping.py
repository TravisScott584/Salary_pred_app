import gzip, shutil

src = "salary_model_r_match.pkl"
dst = "salary_model_r_match.pkl.gz"

with open(src, "rb") as f_in:
    with gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print("✅ Compressed model saved as", dst)
