import os
import openai
from dotenv import load_dotenv
import json
import html
import os, csv, json, html, xml.etree.ElementTree as ET
from query_generator import make_query   # your LLM / template fn


load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

# SYSTEM_PROMPT = (
#     "You are creating short **Italian** questions for a handwriting retrieval task. "
#     "Given a sentence, write ONE question whose answer can be found in that sentence. "
#     "Do NOT quote the sentence verbatim; paraphrase or rephrase. "
#     "Return ONLY the question text.")

SYSTEM_PROMPT = (
    "Sei un assistente che genera domande brevi in italiano per un "
    "benchmark di recupero di documenti manoscritti. "
    "Ti verrà fornita una singola riga trascritta; "
    "scrivi UNA domanda la cui risposta si trova in quella stessa riga. "
    "Non copiare la frase esattamente: parafrasa o riformula. "
    "Rispondi SOLO con la domanda."
)
def make_query(sentence: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Frase: \"{sentence}\""}
        ],
    )
    return resp.choices[0].message.content.strip()

# print(make_query("Non occorre ch'io raccomandi né a voi, né a Giacomo coteste figlie."))
def choose_line(lines, min_len=15):
    """Pick longest line ≥ min_len chars, else absolute longest."""
    candidates = [ln for ln in lines if len(ln) >= min_len] or lines
    return max(candidates, key=len)

def extract_lines(xml_filepath):
    """Return list of text strings from <line ... text="..."/> elements."""
    lines = []
    for _event, elem in ET.iterparse(xml_filepath, events=("end",)):
        if elem.tag.endswith("line") and "text" in elem.attrib:
            lines.append(html.unescape(elem.attrib["text"]))
        elem.clear()
    return lines

def build_manifest(lam_root, out_dir, min_len=15):
    # ensure output directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    qrels_path   = os.path.join(out_dir, "lam_qrels.tsv")
    queries_path = os.path.join(out_dir, "lam_queries.jsonl")

    qrels_f   = open(qrels_path,   "w", encoding="utf-8", newline="")
    queries_f = open(queries_path, "w", encoding="utf-8")
    qrels_w   = csv.writer(qrels_f, delimiter="\t")
    qrels_w.writerow(["query_id", "doc_id", "relevance"])

    qid = 0
    # Walk the tree: lam_root/full_pages/<folder>/xml/*.xml
    full_pages_dir = os.path.join(lam_root, "full_pages")
    print(full_pages_dir)
    count=0
    for root, dirs, files in os.walk(full_pages_dir):
        # print(root)
        if not root.endswith(os.sep + "xml"):
            continue               
        for fname in sorted(files):
            # print(fname)
            if not fname.endswith(".xml"):
                continue
            xml_path = os.path.join(root, fname)          # full path to XML
            # print(xml_path)
            doc_id   = os.path.splitext(fname)[0]         # e.g. "002_02"
            # corresponding image: replace "/xml/" → "/img/" and .xml → .jpg
            img_path = xml_path.replace(os.sep + "xml" + os.sep,
                                        os.sep + "img" + os.sep)[:-4] + ".jpg"
            
            lines = extract_lines(xml_path)
            if not lines:
                continue

            sentence = choose_line(lines, min_len)
            count+=1
            query    = make_query(sentence)

            query_id = f"LAM-{qid:06d}"
            qid += 1

            # write JSONL
            queries_f.write(json.dumps({
                "id":         query_id,
                "query":      query,
                "answer":     sentence,
                "image_path": img_path,
                "doc_id":     doc_id
            }) + "\n")

            # write qrels row
            qrels_w.writerow([query_id, doc_id, 1])
        print(f"writing #{count}")

    qrels_f.close()
    queries_f.close()
    print(f"Generated {qid} queries in '{out_dir}'")
    print(count)

# ─── Example call ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    lam_root = "/data_collection/LAM"        # folder that contains full_pages/
    out_dir  = "queries"                 # output folder
    build_manifest(lam_root, out_dir, min_len=15)
