"""Import the State Farm dataset into Label Studio, fully pre-annotated.

Creates a "State Farm Distracted Driving" project and imports every labeled
train image as a task with its ground-truth class attached as a completed
annotation, so the project opens fully labeled — ready for review, correction,
or extension with your own captures.

Requires Label Studio running with local-file serving pointed at this repo's
data/ directory (so tasks can reference the images in place without uploading
22k files):

    $env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
    $env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT   = "<repo>\\data"
    label-studio start

Authenticates with a Label Studio account (session login — legacy API tokens
are disabled by default in Label Studio >= 1.17):

    python scripts/import_state_farm_labelstudio.py \
        --url http://localhost:8081 --email you@example.com --password ...
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import requests

from prepare_state_farm import CLASS_NAMES  # noqa: E402 - same scripts/ dir

ROOT = Path(__file__).resolve().parent.parent
DRIVER_CSV = ROOT / "data" / "state_farm" / "driver_imgs_list.csv"
BATCH_SIZE = 1000

LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" zoom="true"/>
  <Header value="Driver behavior"/>
  <Choices name="label" toName="image" choice="single" showInline="false">
{choices}
  </Choices>
</View>
""".format(
    choices="\n".join(
        f'    <Choice value="{code}: {name}"/>' for code, name in CLASS_NAMES.items()
    )
)


def login(base_url: str, email: str, password: str) -> requests.Session:
    session = requests.Session()
    login_url = f"{base_url}/user/login/"
    session.get(login_url, timeout=30)
    resp = session.post(
        login_url,
        data={
            "email": email,
            "password": password,
            "csrfmiddlewaretoken": session.cookies.get("csrftoken", ""),
        },
        headers={"Referer": login_url},
        timeout=30,
    )
    resp.raise_for_status()
    whoami = session.get(f"{base_url}/api/current-user/whoami", timeout=30)
    if whoami.status_code != 200:
        raise SystemExit(f"login failed for {email} (HTTP {whoami.status_code})")
    print(f"logged in as {whoami.json()['email']}")
    return session


def post(session: requests.Session, base_url: str, path: str, payload) -> dict:
    resp = session.post(
        f"{base_url}{path}",
        json=payload,
        headers={
            "X-CSRFToken": session.cookies.get("csrftoken", ""),
            "Referer": base_url,
        },
        timeout=300,
    )
    if resp.status_code >= 400:
        raise SystemExit(f"POST {path} -> HTTP {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def build_tasks(rows: list[dict]) -> list[dict]:
    tasks = []
    for row in rows:
        label = f"{row['classname']}: {CLASS_NAMES[row['classname']]}"
        tasks.append(
            {
                "data": {
                    "image": f"/data/local-files/?d=state_farm/train/{row['classname']}/{row['img']}",
                    "subject": row["subject"],
                    "classcode": row["classname"],
                },
                "annotations": [
                    {
                        "was_cancelled": False,
                        "ground_truth": True,
                        "result": [
                            {
                                "from_name": "label",
                                "to_name": "image",
                                "type": "choices",
                                "value": {"choices": [label]},
                            }
                        ],
                    }
                ],
            }
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8081")
    parser.add_argument("--email", default=os.environ.get("LABEL_STUDIO_EMAIL"))
    parser.add_argument("--password", default=os.environ.get("LABEL_STUDIO_PASSWORD"))
    parser.add_argument("--limit", type=int, default=None, help="import only the first N tasks (smoke test)")
    args = parser.parse_args()
    if not args.email or not args.password:
        raise SystemExit("pass --email/--password or set LABEL_STUDIO_EMAIL/LABEL_STUDIO_PASSWORD")

    with open(DRIVER_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    base_url = args.url.rstrip("/")
    session = login(base_url, args.email, args.password)
    project = post(
        session,
        base_url,
        "/api/projects/",
        {
            "title": "State Farm Distracted Driving",
            "description": (
                "22,424 dashboard-camera images across 10 driver-behavior classes "
                "(Kaggle State Farm competition), imported with ground-truth annotations."
            ),
            "label_config": LABEL_CONFIG,
        },
    )
    print(f"created project {project['id']}: {base_url}/projects/{project['id']}")

    tasks = build_tasks(rows)
    for start in range(0, len(tasks), BATCH_SIZE):
        batch = tasks[start : start + BATCH_SIZE]
        post(session, base_url, f"/api/projects/{project['id']}/import", batch)
        print(f"imported {min(start + BATCH_SIZE, len(tasks))}/{len(tasks)}")
    print("done")


if __name__ == "__main__":
    main()
