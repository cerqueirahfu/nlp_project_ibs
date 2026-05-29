# Deploying the Price Pulse Dashboard on EC2 (Amazon Linux 2023)

A step-by-step guide to host your Streamlit dashboard — including the **Amazon
Bedrock** AI features (insight summaries + dataset chat) — on a free-tier
`t3.micro` instance. Assumes the instance is launched, running, and you have
your `.pem` key file and the instance's **public IPv4 address**.

> Reminder: this instance has **no GPU**. Fine-tune your model elsewhere
> (Colab or your laptop), then bring the finished model + results here.

> Order of operations: do the AWS-console prerequisites (steps **A → B → C**)
> first, then the on-instance setup (steps **1 → 9**). The AI smoke test is step **6b**.

---

## 0. Before you start — checklist

- [ ] Instance is **running** (EC2 console shows green "Running")
- [ ] You have your key file (e.g. `ec2pricepulse.pem`)
- [ ] Security group allows inbound **port 22** (SSH) and **port 8501** (Streamlit)
- [ ] You know the instance's **public DNS / IP** (from the EC2 **Connect** screen)
- [ ] **AWS Budget alert** is set (Bedrock is pay-per-use — see step A)
- [ ] **Bedrock model access** confirmed (auto-enabled on first use; Claude may need a one-time use-case form — see step B)
- [ ] EC2 **IAM role** has S3 read + `bedrock:InvokeModel` (see step C)
- [ ] **S3 bucket** created and `results.parquet` uploaded (see step D)

---

## A. Set a budget alert FIRST (do this before anything paid)

Bedrock bills per token. The per-call cost is tiny, but a runaway loop is the
real risk. Protect yourself before writing any AI code.

1. AWS Console → **Billing and Cost Management** → **Budgets** → **Create budget**.
2. Choose **Cost budget**, set a monthly amount (e.g. **$10**).
3. Add an alert threshold at **80%** with your email.
4. (Optional) Create a second budget at **$25** as a hard "stop and investigate" line.

You'll get an email long before you approach your $100. This costs nothing to set up.

---

## B. Enable Bedrock model access (mostly automatic now)

> **Updated:** AWS retired the old "Model access" toggle page. Serverless
> foundation models are now **auto-enabled the first time you invoke them** in
> your account — there's usually nothing to switch on manually.

For **Anthropic (Claude) models specifically**, first-time users may need to
**submit a short use-case form once** before access is granted. Two ways to
handle it:

- **Trigger it via the playground:** Bedrock console → **Model catalog** → find
  **Claude Haiku 4.5** → open in **Playground**. If a use-case form appears, fill
  it in once (e.g. *"Academic project analyzing product-review sentiment"*).
  This enables the model account-wide.
- **Or just run the smoke test (step 6b):** if it returns `OK`, you're already
  enabled — nothing to do. If you get an access/validation error, do the
  playground step above to trigger the form, then retry.

> **Region note:** model availability still varies by region. If the model isn't
> offered in your EC2 region (e.g. some EU regions), point your `boto3` Bedrock
> client at a US region (e.g. `us-east-1`) while keeping the rest of your stack
> where it is.
>
> **Note:** auto-enablement only makes the model *available*. Your EC2 instance
> still needs the **IAM permission** in step C to actually call it — that part
> has not changed.

---

## C. Give the EC2 instance permission for S3 + Bedrock

Attach an **IAM role** to the instance so your code never contains AWS keys.

1. AWS Console → **IAM** → **Roles** → **Create role** → trusted entity **EC2**.
2. Attach permissions. For a student prototype the simplest is the AWS-managed
   policies **`AmazonS3ReadOnlyAccess`** and **`AmazonBedrockFullAccess`**.
   (Tighter: a custom policy allowing only `bedrock:InvokeModel` and
   `s3:GetObject` on your bucket.)
3. Name it e.g. `price-pulse-ec2-role` and create it.
4. EC2 Console → select your instance → **Actions** → **Security** →
   **Modify IAM role** → choose `price-pulse-ec2-role` → **Update**.

No restart needed — the role is available to the instance immediately.

---

## D. Create the S3 bucket and upload your results

Do this from your **laptop** (where the results file is). You need the AWS CLI
installed locally and configured once with `aws configure` (paste your access
keys from IAM → Security credentials).

1. **Create the bucket** (bucket names are globally unique — pick your own and
   use it everywhere below):

   ```bash
   aws s3 mb s3://pricepulse-data --region us-east-1
   ```

   > Use the **same region** as your EC2 instance where possible. If your EC2 is
   > in an EU region but you call Bedrock in `us-east-1`, that's fine — S3 just
   > needs to be reachable; same-region as EC2 is simplest and avoids transfer
   > quirks.

2. **Upload your processed results — BOTH parquet files** (the app needs each
   one: `results.parquet` drives the dashboard aggregates, `products.parquet`
   drives the Product drill-down page):

   ```bash
   aws s3 cp results.parquet s3://pricepulse-data/results.parquet
   aws s3 cp products.parquet s3://pricepulse-data/products.parquet
   ```

3. **(Optional) upload the raw dataset too**, if you want everything in one place:

   ```bash
   aws s3 cp data/amazon.csv s3://pricepulse-data/raw/amazon.csv
   ```

4. **Verify they're there:**

   ```bash
   aws s3 ls s3://pricepulse-data/
   ```

**Cost note:** at this size (a few MB) S3 is effectively free — well within the
5 GB free tier, and pennies even beyond it. Keep the bucket **private** (the
default); the EC2 IAM role from step C is what grants read access, so you never
need to make it public.

> **Re-upload whenever results change:** every time you re-run the ABSA pipeline
> on your laptop/Colab and get a new `results.parquet`, repeat step D.2 to push
> the new version up. The instance picks it up next time it downloads (step 6).

---

## 1. Fix key file permissions (local machine)

SSH refuses to use a key file that others can read. Run this once, on your own
computer, in the folder where the `.pem` lives:

```bash
chmod 400 ec2pricepulse.pem
```

> **On Windows (PowerShell)** there is no `chmod`. Use `icacls` instead, run
> from the folder containing the key:
>
> ```powershell
> icacls.exe ec2pricepulse.pem /reset
> icacls.exe ec2pricepulse.pem /grant:r "$($env:USERNAME):(R)"
> icacls.exe ec2pricepulse.pem /inheritance:r
> ```
>
> If the key sits in a OneDrive folder and SSH still complains, copy it to a
> plain local folder (e.g. `C:\Users\<you>\.ssh\`) and run `icacls` there.

---

## 2. Connect to the instance

```bash
ssh -i ec2pricepulse.pem ec2-user@<PUBLIC_IP>
```

Replace `<PUBLIC_IP>` with your address. Type `yes` when asked to trust the host.
The username is **`ec2-user`** for Amazon Linux 2023 (it was on your launch screen).

---

## 3. Update the system and install Python tools

Amazon Linux 2023 ships with Python 3.9+. Update packages and install `pip`
and `git`:

```bash
sudo dnf update -y
sudo dnf install -y python3 python3-pip git
```

Verify:

```bash
python3 --version
pip3 --version
```

---

## 4. Get your project onto the instance

Pick **one** of these.

### Option A — clone from GitHub (recommended)

```bash
git clone https://github.com/<your-username>/price-pulse.git
cd price-pulse
```

### Option B — copy from your laptop with scp

Run this **on your laptop**, not on the instance:

```bash
scp -i ec2pricepulse.pem -r ./price-pulse ec2-user@<PUBLIC_IP>:~/
```

Then back on the instance: `cd price-pulse`

---

## 5. Set up a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, at minimum:

```bash
pip install streamlit pandas pyarrow boto3
```

(`pyarrow` lets pandas read Parquet; `boto3` is for **both** S3 and Bedrock —
the same library provides the `bedrock-runtime` client, so no extra package is
needed for the AI features.)

---

## 6. Bring your results onto the instance (from S3)

Run this **on the instance** (in your SSH/VS Code terminal where the prompt
shows `ec2-user@ip-...`), not on your laptop. Download **both** parquet files
into the **project root** (alongside `app.py`) — that's exactly where the app
looks for them by default (`utils/helpers.py`), so no env vars are needed:

```bash
cd ~/price-pulse
aws s3 cp s3://pricepulse-data/results.parquet ./results.parquet
aws s3 cp s3://pricepulse-data/products.parquet ./products.parquet
```

This works with no credentials because the **IAM role from step C** grants the
instance S3 read access. Confirm they arrived:

```bash
ls -lh results.parquet products.parquet
```

> Replace `pricepulse-data` with whatever bucket name you actually created in
> step D. If you get `AccessDenied`, the IAM role is missing/not attached
> (revisit step C). If you get `NoSuchBucket`, check the name and region.

> **Alternative — read straight from S3 (no download):** the app also accepts
> `RESULTS_S3_URI` / `PRODUCTS_S3_URI` env vars pointing at `s3://...` paths.
> That route needs `s3fs` installed (`pip install s3fs`). Downloading to disk as
> above is simpler and is what this guide assumes.

**Whenever you update results:** re-upload both files from your laptop (step
D.2), then re-run these `aws s3 cp` commands on the instance to pull the new
versions.

---

## 6b. Verify Bedrock access (quick smoke test)

Before launching the app, confirm the instance can actually reach Bedrock with
its IAM role. With the venv active, run this one-off Python check (replace the
`region_name` if you enabled the model in a US region):

```bash
python3 - <<'EOF'
import boto3, json
client = boto3.client("bedrock-runtime", region_name="us-east-1")
body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
}
resp = client.invoke_model(
    modelId="anthropic.claude-haiku-4-5-20251001-v1:0",  # adjust to the exact ID shown in Bedrock console
    body=json.dumps(body),
)
print(json.loads(resp["body"].read())["content"][0]["text"])
EOF
```

- If it prints `OK` (or similar) — Bedrock is wired up correctly.
- `AccessDeniedException` → the IAM role is missing `bedrock:InvokeModel` (step C).
- `ValidationException` / model-not-found → wrong `modelId` or model access not
  enabled in this region (step B). Copy the exact model ID from the Bedrock console.

> Note the **exact `modelId`** that works here and use the same string in your
> app's `ai/bedrock_client.py`. Always keep `max_tokens` capped, as in this test.

---

## 7. Test that it runs

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

The `--server.address 0.0.0.0` part is **essential** — it tells Streamlit to
accept connections from outside the instance. Without it, only the instance
itself can reach the app.

> **Important:** Streamlit will print `URL: http://0.0.0.0:8501`. **Do NOT open
> that in your browser** — `0.0.0.0` is a "listen on all interfaces" address,
> not a real destination. It will fail with `ERR_ADDRESS_INVALID`. Use your
> instance's **public DNS or IP** instead (next).

Now open a browser on your own machine — use the **public DNS** from the EC2
Connect screen:

```
http://ec2-XX-XX-XX-XX.compute-1.amazonaws.com:8501
```

(or `http://<PUBLIC_IP>:8501`). Note it's **`http://`** not `https://`, and the
**`:8501`** suffix is required.

If the page loads — success. If it spins forever or won't connect, your security
group is almost certainly missing the **port 8501** inbound rule (see
troubleshooting). The public DNS/IP also **changes on every stop/start** — grab
the current one from the Connect screen each session.

Press `Ctrl+C` to stop it for now.

---

## 8. Keep the app running after you disconnect

If you just run `streamlit run`, the app dies the moment you close SSH. Use a
`tmux` session so it survives.

```bash
sudo dnf install -y tmux
tmux new -s dashboard
```

Inside the tmux session, start the app:

```bash
source venv/bin/activate
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Now **detach** without stopping it: press `Ctrl+B`, then `D`.

You can safely close SSH — the dashboard keeps running.

To come back later:

```bash
ssh -i ec2pricepulse.pem ec2-user@<PUBLIC_IP>
tmux attach -t dashboard
```

To stop the app: attach, then `Ctrl+C`.

---

## 9. When you're done for the day — STOP the instance

In the EC2 console: select the instance → **Instance state** → **Stop**.

- **Stop** = compute billing pauses, disk + setup preserved. Use this daily.
- **Terminate** = deletes everything. Only when the project is fully finished.

> Note: a stopped instance gets a **new public IP** when you start it again.
> If you want a permanent address, allocate an **Elastic IP** (free while
> attached to a running instance). For a student demo, just grab the new IP
> each time — simpler.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Browser hangs / can't connect on :8501 | Security group missing port 8501 | Add inbound rule: Custom TCP, port 8501, source `0.0.0.0/0` (or your IP) |
| `Permission denied (publickey)` on SSH | Wrong key, wrong user, or key perms | Use `ec2-user`, check `chmod 400`, confirm right `.pem` |
| App works in SSH but dies after logout | Not using tmux | Run inside tmux (step 8) |
| `streamlit: command not found` | venv not activated | `source venv/bin/activate` |
| S3 `AccessDenied` | No IAM role / wrong creds | Attach S3-read IAM role to instance (step C) |
| Page loads only on `localhost` | Missing `--server.address 0.0.0.0` | Add that flag |
| Browser shows `ERR_ADDRESS_INVALID` on `0.0.0.0:8501` | Tried to open the literal `0.0.0.0` URL Streamlit printed | Use the public DNS/IP instead, e.g. `http://<public-dns>:8501` |
| Bedrock `AccessDeniedException` | IAM role lacks `bedrock:InvokeModel` | Add Bedrock permission to the role (step C) |
| Bedrock `ValidationException` / model not found | Wrong `modelId`, region, or Claude use-case form not yet submitted | Check exact ID from Model catalog; if Claude, open it in Playground once to submit the use-case form (step B) |
| Bedrock works locally but not on EC2 | Different region than where model is enabled | Set `region_name` to the region with model access |
| AI text empty but no error | `max_tokens` too low or response parsing | Raise cap slightly; check you read `content[0]["text"]` |

---

## Quick reference — daily workflow

```bash
# Connect
ssh -i ec2pricepulse.pem ec2-user@<PUBLIC_IP>

# Resume the dashboard
tmux attach -t dashboard
# ...do stuff, then Ctrl+B then D to detach

# Done for the day: Stop the instance in the EC2 console
```

> **Cost hygiene:** stopping the EC2 instance pauses *compute* charges. Bedrock
> only bills when the app actually calls it (button click / chat submit), so a
> stopped instance makes zero Bedrock calls. Keep the budget alert from step A
> active as your safety net throughout the project.

---

## Optional: auto-start on reboot (nice-to-have)

If you want the dashboard to launch automatically whenever the instance boots,
create a small systemd service. Skip this for a basic prototype — tmux is
enough — but here it is if you want it:

```bash
sudo tee /etc/systemd/system/pricepulse.service > /dev/null <<'EOF'
[Unit]
Description=Price Pulse Streamlit Dashboard
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/price-pulse
ExecStart=/home/ec2-user/price-pulse/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now pricepulse
```

Check status with `sudo systemctl status pricepulse`.
