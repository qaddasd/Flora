/* E2E test for the Flora web demo using system Chrome via puppeteer-core.
 *
 * Usage: node e2e_test.js [--shots-only]
 * - opens http://127.0.0.1:8765/index.html?autotest=1
 * - waits for the #autotest-result report, prints it
 * - exercises UI: template playback, Open/Close, Inflated, Compare tab
 * - saves screenshots to D:\Flora-Ai\tools\shots\
 */

const puppeteer = require("puppeteer-core");
const fs = require("fs");
const path = require("path");

const BASE = "http://127.0.0.1:8765";
const SHOTS = path.join(__dirname, "shots");

(async () => {
  fs.mkdirSync(SHOTS, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
    headless: "new",
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan",
      "--use-webgpu-adapter=default",
      "--window-size=1560,940",
      "--mute-audio",
      "--autoplay-policy=no-user-gesture-required",
    ],
    defaultViewport: { width: 1560, height: 940 },
  });

  const page = await browser.newPage();
  const logs = [];
  page.on("console", (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on("pageerror", (e) => logs.push(`[pageerror] ${e.message}`));

  console.log("== open autotest page ==");
  await page.goto(`${BASE}/index.html?autotest=1`, { waitUntil: "domcontentloaded" });

  // wait for self-test result (fusion forward pass can take a while: model DL)
  try {
    await page.waitForSelector("#autotest-result", { timeout: 240000 });
    const report = await page.$eval("#autotest-result", (el) => el.textContent);
    console.log("---- SELFTEST REPORT ----");
    console.log(report);
    console.log("-------------------------");
  } catch (e) {
    console.log("!! selftest did not finish in time");
  }
  await page.screenshot({ path: path.join(SHOTS, "01_initial.png") });

  // template playback
  console.log("== click template card ==");
  const cards = await page.$$("#examples-grid .video-card:not(.upload-card)");
  if (cards.length) {
    await cards[0].click();
    await new Promise((r) => setTimeout(r, 3500));
  }
  await page.screenshot({ path: path.join(SHOTS, "02_template_playing.png") });

  // Open hemispheres
  await page.click('#seg-open button[data-open="1"]');
  await new Promise((r) => setTimeout(r, 1600));
  await page.screenshot({ path: path.join(SHOTS, "03_open.png") });

  // Inflated
  await page.click('#seg-surface button[data-surface="inflated"]');
  await new Promise((r) => setTimeout(r, 1600));
  await page.screenshot({ path: path.join(SHOTS, "04_inflated_open.png") });

  // back to normal + closed
  await page.click('#seg-surface button[data-surface="normal"]');
  await page.click('#seg-open button[data-open="0"]');

  // Compare tab + dataset clip
  console.log("== compare tab ==");
  await page.click('[data-tab="compare"]');
  await new Promise((r) => setTimeout(r, 400));
  const items = await page.$$(".compare-item");
  if (items.length) {
    await items[0].click();
    await new Promise((r) => setTimeout(r, 3000));
  }
  await page.screenshot({ path: path.join(SHOTS, "05_compare.png") });

  // In-Silico: lesion audio modality on a template clip (live fusion rerun)
  console.log("== in-silico tab ==");
  await page.click('[data-tab="insilico"]');
  await new Promise((r) => setTimeout(r, 300));
  await page.click('#is-mods .chip[data-mod="audio"]'); // lesion audio
  await page.click("#is-run");
  try {
    await page.waitForSelector("#is-result:not(.hidden)", { timeout: 180000 });
    const delta = await page.$eval("#is-delta", (el) => el.textContent);
    console.log("in-silico mean|Δ| =", delta);
    await new Promise((r) => setTimeout(r, 2500));
  } catch {
    console.log("!! in-silico run did not finish");
  }
  await page.screenshot({ path: path.join(SHOTS, "06_insilico.png") });

  // Upload pipeline: real video → decode → MobileViT → Whisper mel → fusion
  console.log("== upload pipeline ==");
  await page.click('[data-tab="browse"]');
  await new Promise((r) => setTimeout(r, 300));
  await page.click("#examples-grid .upload-card");
  await new Promise((r) => setTimeout(r, 300));
  const input = await page.$("#um-input");
  await input.uploadFile(path.join(__dirname, "..", "web-demo", "videos", "ocean_waves.mp4"));
  try {
    await page.waitForSelector("#um-view:not(.hidden)", { timeout: 420000 });
    await page.screenshot({ path: path.join(SHOTS, "07_upload_pipeline.png") });
    await page.click("#um-view");
    await new Promise((r) => setTimeout(r, 4000));
    console.log("upload pipeline: OK");
  } catch {
    console.log("!! upload pipeline did not finish");
  }
  await page.screenshot({ path: path.join(SHOTS, "08_upload_result.png") });

  console.log("== console output (last 40) ==");
  logs.slice(-40).forEach((l) => console.log(l));

  await browser.close();
  console.log("DONE");
})().catch((e) => { console.error("E2E FAILED:", e); process.exit(1); });
