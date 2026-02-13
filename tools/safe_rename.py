import tokenize
import io

def rename_token(token_text):
    if token_text == "time":
        return "sam_time_ref"
    if token_text == "datetime":
        return "sam_datetime_ref"
    return token_text

with open("src/python/complete_sam_unified.py", "r") as f:
    content = f.read()

result = []
tokens = tokenize.generate_tokens(io.StringIO(content).readline)
for toknum, tokval, _, _, _ in tokens:
    if toknum == tokenize.NAME:
        result.append((toknum, rename_token(tokval)))
    else:
        result.append((toknum, tokval))

new_content = tokenize.untokenize(result)
with open("src/python/complete_sam_unified.py", "w") as f:
    f.write(new_content)
print("Safe rename complete.")
