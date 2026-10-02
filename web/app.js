
"use strict";

// ============================================
// 1. CONFIGURATION
// ============================================

const CONFIG = {
  modelUrl: "./models/model.onnx",
  nsfwModelUrl: "./models/nsfwjs/model.json",
  inputWidth: 224,
  inputHeight: 224,
  resizeSize: 256,
  meanBGR: [104, 117, 123]
};

const CDN =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

// Both JS and WASM binaries must match versions.
if (typeof ort !== "undefined") {
  ort.env.wasm.wasmPaths = CDN;
  ort.env.wasm.numThreads = 1;
}

const TFJS_CDN =
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/";

const CLASS_NAMES = {
  opennsfw2: ["SFW", "NSFW"],
  nsfwjs: ["Drawing", "Hentai", "Neutral", "Porn", "Sexy"]
};


// ============================================
// 2. STATE
// ============================================

const $ = id => document.getElementById(id);

const fileInput = $("fileInput");
const selectButton = $("selectButton");
const executionMode = $("executionMode");
const modelChoice = $("modelChoice");
const preview = $("preview");
const modelStatus = $("modelStatus");
const providerBadge = $("provider");
const errorEl = $("error");

const sessions = {
  opennsfw2: { wasm: null, webgpu: null },
  nsfwjs: { wasm: null, webgpu: null }
};

let session = null;
let currentMode = "wasm";
let currentModel = "opennsfw2";

let isRunning = false;
let isLoading = false;
let previewUrl = null;
let selectedFile = null;
let selectedFolder = null;
let isBatchRunning = false;
let batchCancelRequested = false;

const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];

// Keep measurements separate for each model and device.
const benchmarks = {
  opennsfw2: { wasm: null, webgpu: null },
  nsfwjs: { wasm: null, webgpu: null }
};


// ============================================
// 3. UI HELPERS
// ============================================

function setStatus(message) {
  modelStatus.textContent = message;
}

function providerLabel(model, mode) {
  if (mode === "wasm") return "CPU / WASM";
  return model === "opennsfw2" ? "WebGPU + WASM" : "WebGPU";
}

function setProvider(model, mode) {
  providerBadge.textContent = providerLabel(model, mode);
  $("gpuBenchmarkLabel").textContent = providerLabel(model, "webgpu");
  $("deviceHint").textContent = model === "opennsfw2"
    ? "CPU runs using WASM. WebGPU prefers GPU execution with optional WASM operator fallback."
    : "CPU runs using TensorFlow.js WASM. WebGPU uses the TensorFlow.js WebGPU backend.";
  $("benchmarkTimeLabel").textContent = model === "nsfwjs"
    ? "Classify time (includes preprocessing)"
    : "Inference time";
}

function showError(error) {
  console.error(error);

  errorEl.style.display = "block";

  errorEl.textContent =
    error instanceof Error
      ? error.message
      : String(error);
}

function clearError() {
  errorEl.style.display = "none";
  errorEl.textContent = "";
}

function updateControls() {
  const busy = isRunning || isLoading;

  selectButton.disabled = busy || !session;
  executionMode.disabled = busy;
  modelChoice.disabled = busy;
  $("folderButton").disabled = busy || !session;
  $("batchRunButton").disabled = busy || !session || !selectedFolder;
  $("batchCancelButton").hidden = !isBatchRunning;
  $("batchCancelButton").disabled = batchCancelRequested;
}

function formatMs(value) {
  return value.toFixed(2) + " ms";
}

function nextFrame() {
  return new Promise(resolve => {
    requestAnimationFrame(() => resolve());
  });
}


// ============================================
// 4. GPU SUPPORT
// ============================================

async function checkWebGPU() {
  if (!navigator.gpu) {
    throw new Error(
      "WebGPU is not supported by this browser."
    );
  }

  const adapter =
    await navigator.gpu.requestAdapter();

  if (!adapter) {
    throw new Error(
      "No WebGPU adapter is available."
    );
  }

  return true;
}


// ============================================
// 5. MODEL WARMUP
// ============================================

async function warmup(targetSession) {
  const data = new Float32Array(
    CONFIG.inputWidth *
    CONFIG.inputHeight *
    3
  );

  const tensor = new ort.Tensor(
    "float32",
    data,
    [1, 224, 224, 3]
  );

  let outputs = null;

  try {
    outputs = await targetSession.run({
      [targetSession.inputNames[0]]: tensor
    });

    console.log("Model warmup completed");

  } finally {
    tensor.dispose();

    if (outputs) {
      for (const output of Object.values(outputs)) {
        if (typeof output.dispose === "function") {
          output.dispose();
        }
      }
    }
  }
}


// ============================================
// 6. CREATE MODEL SESSION
// ============================================

async function createSession(mode) {
  if (typeof ort === "undefined") {
    throw new Error("ONNX Runtime failed to load. Reload the page to retry.");
  }
  const providers =
    mode === "webgpu"
      ? ["webgpu", "wasm"]
      : ["wasm"];

  console.log(
    "Creating session:",
    providers
  );

  const newSession =
    await ort.InferenceSession.create(
      CONFIG.modelUrl,
      {
        executionProviders: providers,
        graphOptimizationLevel: "all"
      }
    );

  try {
    // Execute once before accepting the session.
    await warmup(newSession);

    return newSession;

  } catch (error) {
    await newSession.release();
    throw error;
  }
}


// ============================================
// 7. SWITCH EXECUTION PROVIDER
// ============================================

async function setNsfwBackend(mode) {
  if (typeof tf === "undefined" || typeof nsfwjs === "undefined") {
    throw new Error("NSFWJS or TensorFlow.js failed to load. Reload the page to retry.");
  }
  if (mode === "wasm") {
    if (!tf.wasm) {
      throw new Error("TensorFlow.js WASM backend failed to load.");
    }
    tf.wasm.setWasmPaths(TFJS_CDN);
    tf.env().set("WASM_HAS_MULTITHREAD_SUPPORT", false);
  }
  if (!await tf.setBackend(mode)) {
    throw new Error(`TensorFlow.js ${mode} backend is unavailable.`);
  }
  await tf.ready();
}

async function createNsfwSession() {
  const model = await nsfwjs.load(CONFIG.nsfwModelUrl);
  try {
    const canvas = createCanvas(224, 224);
    canvas.getContext("2d").fillRect(0, 0, 224, 224);
    await model.classify(canvas, 5);
    return model;
  } catch (error) {
    model.dispose();
    throw error;
  }
}

async function loadModel(model = "opennsfw2", mode = "wasm") {
  if (isRunning || isLoading) return false;

  isLoading = true;
  updateControls();
  clearError();
  setStatus(`Loading ${model} (${providerLabel(model, mode)})...`);

  try {
    if (mode === "webgpu") await checkWebGPU();
    // TFJS has one global backend: reactivate it even for cached models.
    if (model === "nsfwjs") await setNsfwBackend(mode);

    if (!sessions[model][mode]) {
      sessions[model][mode] = model === "nsfwjs"
        ? await createNsfwSession()
        : await createSession(mode);
    }

    session = sessions[model][mode];
    currentModel = model;
    currentMode = mode;
    modelChoice.value = model;
    executionMode.value = mode;
    setProvider(model, mode);
    setStatus(`${model} (${providerLabel(model, mode)}) ready`);
    $("results").style.display = "none";
    $("classResults").replaceChildren();
    updateBenchmarkTable();
    return true;
  } catch (error) {
    // Restore both the selection and TFJS's global backend after a failed switch.
    if (session && currentModel === "nsfwjs") {
      try {
        await setNsfwBackend(currentMode);
      } catch (restoreError) {
        session = null;
        showError(new Error(`${error.message}\nBackend restore failed: ${restoreError.message}`));
      }
    }
    if (!errorEl.textContent) showError(error);
    modelChoice.value = currentModel;
    executionMode.value = currentMode;
    setStatus(session
      ? "Model/device switch failed. Previous model ready."
      : "Model initialization failed");
    return false;
  } finally {
    isLoading = false;
    updateControls();
  }
}


// ============================================
// 8. CANVAS HELPERS
// ============================================

function createCanvas(width, height) {
  const canvas = document.createElement("canvas");

  canvas.width = width;
  canvas.height = height;

  return canvas;
}

function canvasToBlob(canvas, type, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      blob => {
        if (blob) {
          resolve(blob);
        } else {
          reject(
            new Error("Canvas encoding failed")
          );
        }
      },
      type,
      quality
    );
  });
}


// ============================================
// 9. IMAGE PREPROCESSING
// ============================================
//
// OpenNSFW2 YAHOO preprocessing:
//
// Resize 256x256
// JPEG encode/decode
// Center crop 224x224
// RGB -> BGR
// Subtract [104,117,123]
// Return float32 NHWC
//
// ============================================

async function preprocessImage(file) {
  let originalBitmap = null;
  let jpegBitmap = null;

  try {
    // Decode image.
    originalBitmap =
      await createImageBitmap(file);

    // Resize to 256x256.
    const resizeCanvas = createCanvas(
      CONFIG.resizeSize,
      CONFIG.resizeSize
    );

    const resizeCtx =
      resizeCanvas.getContext("2d");

    resizeCtx.imageSmoothingEnabled = true;
    resizeCtx.imageSmoothingQuality = "high";

    resizeCtx.drawImage(
      originalBitmap,
      0,
      0,
      CONFIG.resizeSize,
      CONFIG.resizeSize
    );

    // JPEG round-trip.
    const jpegBlob = await canvasToBlob(
      resizeCanvas,
      "image/jpeg",
      0.75
    );

    jpegBitmap =
      await createImageBitmap(jpegBlob);

    // Center crop 224x224.
    const cropCanvas = createCanvas(
      CONFIG.inputWidth,
      CONFIG.inputHeight
    );

    const cropCtx = cropCanvas.getContext(
      "2d",
      { willReadFrequently: true }
    );

    const cropOffset =
      (CONFIG.resizeSize - CONFIG.inputWidth) / 2;

    cropCtx.drawImage(
      jpegBitmap,

      cropOffset,
      cropOffset,

      CONFIG.inputWidth,
      CONFIG.inputHeight,

      0,
      0,

      CONFIG.inputWidth,
      CONFIG.inputHeight
    );

    const pixels = cropCtx.getImageData(
      0,
      0,
      CONFIG.inputWidth,
      CONFIG.inputHeight
    ).data;

    // Create NHWC tensor.
    const totalPixels =
      CONFIG.inputWidth *
      CONFIG.inputHeight;

    const tensorData =
      new Float32Array(totalPixels * 3);

    for (let i = 0; i < totalPixels; i++) {
      const src = i * 4;
      const dst = i * 3;

      const r = pixels[src];
      const g = pixels[src + 1];
      const b = pixels[src + 2];

      // BGR - mean.
      tensorData[dst] =
        b - CONFIG.meanBGR[0];

      tensorData[dst + 1] =
        g - CONFIG.meanBGR[1];

      tensorData[dst + 2] =
        r - CONFIG.meanBGR[2];
    }

    return new ort.Tensor(
      "float32",
      tensorData,
      [
        1,
        CONFIG.inputHeight,
        CONFIG.inputWidth,
        3
      ]
    );

  } finally {
    if (originalBitmap) {
      originalBitmap.close();
    }

    if (jpegBitmap) {
      jpegBitmap.close();
    }
  }
}


// ============================================
// 10. RUN INFERENCE
// ============================================

// Shared prediction path for single images and folders. Owns temporary resources.
async function predictImage(file, activeSession, activeModel, imageElement = null) {
  let inputTensor = null;
  let outputs = null;
  let imageUrl = null;
  const totalStart = performance.now();
  try {
    const preprocessStart = performance.now();
    if (activeModel === "nsfwjs") {
      if (!imageElement) {
        imageElement = new Image();
        imageUrl = URL.createObjectURL(file);
        imageElement.src = imageUrl;
      }
      await imageElement.decode();
    } else {
      inputTensor = await preprocessImage(file);
    }
    const preprocessMs = performance.now() - preprocessStart;
    const inferenceStart = performance.now();
    let predictions;
    if (activeModel === "nsfwjs") {
      const raw = await activeSession.classify(imageElement, 5);
      predictions = CLASS_NAMES.nsfwjs.map(className => {
        const prediction = raw.find(item => item.className === className);
        if (!prediction) throw new Error(`Missing model class: ${className}`);
        return prediction;
      });
    } else {
      outputs = await activeSession.run({
        [activeSession.inputNames[0]]: inputTensor
      });
      const scores = await outputs[activeSession.outputNames[0]].getData();
      if (scores.length !== CLASS_NAMES.opennsfw2.length) {
        throw new Error("Unexpected model output length: " + scores.length);
      }
      predictions = CLASS_NAMES.opennsfw2.map((className, index) => ({
        className, probability: Number(scores[index])
      }));
    }
    const inferenceMs = performance.now() - inferenceStart;
    const totalMs = performance.now() - totalStart;
    for (const { className, probability } of predictions) {
      if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
        throw new Error(`Invalid probability for ${className}`);
      }
    }
    return { predictions, preprocessMs, inferenceMs, totalMs };
  } finally {
    if (inputTensor) inputTensor.dispose();
    if (outputs) {
      for (const output of Object.values(outputs)) {
        if (typeof output.dispose === "function") output.dispose();
      }
    }
    if (imageUrl) URL.revokeObjectURL(imageUrl);
  }
}

async function runInference(file) {
  if (!session || isRunning || isLoading) return;

  isRunning = true;
  updateControls();
  clearError();
  const activeSession = session;
  const activeMode = currentMode;
  const activeModel = currentModel;
  try {
    $("results").style.display = "none";
    setStatus(`Running ${activeModel} (${providerLabel(activeModel, activeMode)})...`);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.style.display = "block";
    await nextFrame();
    const result = await predictImage(file, activeSession, activeModel, preview);
    benchmarks[activeModel][activeMode] = result.inferenceMs;
    updateBenchmarkTable();
    showResults({ ...result, model: activeModel, mode: activeMode });
    setStatus("Inference completed");
  } catch (error) {
    showError(error);
    setStatus("Inference failed");
  } finally {
    isRunning = false;
    updateControls();
  }
}


// ============================================
// 11. SHOW RESULTS
// ============================================

function showResults({ predictions, preprocessMs, inferenceMs, totalMs, model, mode }) {
  const rows = predictions.map(({ className, probability }) => {
    const metric = document.createElement("div");
    metric.className = "metric";
    const header = document.createElement("div");
    header.className = "metric-header";
    const label = document.createElement("span");
    label.textContent = className + ":";
    const value = document.createElement("span");
    value.textContent = (probability * 100).toFixed(1) + "%";
    header.append(label, value);
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("div");
    fill.className = "fill";
    fill.dataset.class = className;
    fill.style.width = Math.max(0, Math.min(100, probability * 100)) + "%";
    bar.append(fill);
    metric.append(header, bar);
    return metric;
  });
  $("classResults").replaceChildren(...rows);
  $("executionProvider").textContent = providerLabel(model, mode);
  $("preprocessLabel").textContent = model === "nsfwjs" ? "Image decoding" : "Preprocessing";
  $("inferenceLabel").textContent = model === "nsfwjs"
    ? "Classify (includes preprocessing)" : "Inference";
  $("preprocessTime").textContent = formatMs(preprocessMs);
  $("inferenceTime").textContent = formatMs(inferenceMs);
  $("totalTime").textContent = formatMs(totalMs);
  $("results").style.display = "block";
}


// ============================================
// 12. BENCHMARK TABLE
// ============================================

function updateBenchmarkTable() {
  const times = benchmarks[currentModel];
  $("cpuBenchmark").textContent = times.wasm === null ? "-" : formatMs(times.wasm);
  $("gpuBenchmark").textContent = times.webgpu === null ? "-" : formatMs(times.webgpu);
}


// ============================================
// 13. FOLDER STATISTICS AND CHARTS
// ============================================

function probabilityBin(probability) {
  // [0,10), [10,20), ... [90,100], including exactly 100% in the last bin.
  return Math.min(9, Math.floor(probability * 10));
}

function binLabel(index) {
  return `${index * 10}–${(index + 1) * 10}%`;
}

function mostCommonRanges(bins) {
  const largest = Math.max(...bins);
  return largest === 0 ? "—" : bins
    .map((count, index) => count === largest ? binLabel(index) : null)
    .filter(Boolean).join(", ");
}

function recordPrediction(stats, predictions) {
  // Preserve class order for exact ties.
  const winner = predictions.reduce((best, item) =>
    item.probability > best.probability ? item : best);
  const group = stats.classes[winner.className];
  group.count++;
  group.bins[probabilityBin(winner.probability)]++;
  stats.succeeded++;
}

function svgNode(tag, attributes = {}, content = null) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (content !== null) node.textContent = content;
  return node;
}

function createHistogram(className, group, scaleMax) {
  const card = document.createElement("div");
  card.className = "histogram-card";
  const heading = document.createElement("h4");
  heading.textContent = `${className} · ${group.count} images`;
  const chart = svgNode("svg", {
    viewBox: "0 0 400 215", role: "img",
    "aria-label": `${className}: image counts by predicted probability, from 0 to 100 percent`
  });
  chart.append(svgNode("title", {}, `${className} probability distribution`));
  const left = 40, top = 20, width = 344, height = 136;
  const binWidth = width / 10;
  const baseline = top + height;
  for (let tick = 0; tick <= 2; tick++) {
    const y = baseline - tick * height / 2;
    chart.append(svgNode("line", { x1: left, x2: left + width, y1: y, y2: y, stroke: "#e2e8f0" }));
    chart.append(svgNode("text", { x: left - 8, y: y + 4, "text-anchor": "end", "font-size": 14, fill: "#64748b" }, String(tick * scaleMax / 2)));
  }
  group.bins.forEach((count, index) => {
    const barHeight = count / scaleMax * height;
    const rect = svgNode("rect", {
      x: left + index * binWidth + 3, y: baseline - barHeight,
      width: binWidth - 6, height: barHeight, class: "histogram-bar", tabindex: "0",
      "aria-label": `${binLabel(index)}: ${count} images`
    });
    rect.append(svgNode("title", {}, `${binLabel(index)}: ${count} images`));
    chart.append(rect);
    if (count) chart.append(svgNode("text", {
      x: left + (index + 0.5) * binWidth, y: baseline - barHeight - 4,
      "text-anchor": "middle", "font-size": 14, fill: "#111827"
    }, String(count)));
  });
  for (let tick = 0; tick <= 10; tick++) {
    chart.append(svgNode("text", {
      x: left + tick * binWidth, y: baseline + 22,
      "text-anchor": "middle", "font-size": 14, fill: "#64748b"
    }, String(tick * 10)));
  }
  chart.append(svgNode("text", { x: left + width / 2, y: 205, "text-anchor": "middle", "font-size": 14, fill: "#64748b" }, "Predicted class probability (%)"));
  const common = document.createElement("p");
  common.className = "hint";
  common.textContent = group.count
    ? `Most common range: ${mostCommonRanges(group.bins)}`
    : "No images assigned to this class.";
  card.append(heading, chart, common);
  return card;
}

function renderBatchReport(stats) {
  $("batchReport").hidden = false;
  $("batchReportTitle").textContent = `${stats.folderName} · ${stats.model} · ${providerLabel(stats.model, stats.mode)}`;
  const outcome = stats.stopped ? "Stopped" : stats.completed === stats.total ? "Completed" : "Processing";
  $("batchSummary").textContent = `${outcome}: ${stats.completed}/${stats.total} images processed. ` +
    `${stats.succeeded} classified, ${stats.failed} failed, ${stats.ignored} unsupported files skipped. ` +
    "Class percentages are shares of classified images.";
  const rows = Object.entries(stats.classes).map(([className, group]) => {
    const row = document.createElement("tr");
    const share = stats.succeeded ? (group.count / stats.succeeded * 100).toFixed(1) + "%" : "—";
    for (const text of [className, String(group.count), share, mostCommonRanges(group.bins)]) {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.append(cell);
    }
    return row;
  });
  $("batchClassRows").replaceChildren(...rows);
  // Use the same integer Y scale for all class charts in this run.
  const largest = Math.max(1, ...Object.values(stats.classes).flatMap(group => group.bins));
  const scaleMax = Math.max(2, Math.ceil(largest / 2) * 2);
  $("batchCharts").replaceChildren(...Object.entries(stats.classes)
    .map(([className, group]) => createHistogram(className, group, scaleMax)));
}

function updateBatchProgress(stats, file = null) {
  $("batchProgress").max = stats.total;
  $("batchProgress").value = stats.completed;
  const percent = (stats.completed / stats.total * 100).toFixed(1);
  $("batchProgressText").textContent = `${stats.stopped ? "Stopped" : "Processed"} ${stats.completed}/${stats.total} images (${percent}%)` +
    (file ? ` · ${file.webkitRelativePath || file.name}` : "");
}

async function runBatch() {
  if (!session || isRunning || isLoading || !selectedFolder) return;
  isRunning = true;
  isBatchRunning = true;
  batchCancelRequested = false;
  updateControls();
  clearError();
  $("results").style.display = "none";
  preview.style.display = "none";
  $("batchProgressPanel").hidden = false;
  $("batchErrorDetails").hidden = true;
  $("batchErrors").replaceChildren();
  const activeSession = session;
  const folder = selectedFolder;
  const stats = {
    folderName: folder.name, model: currentModel, mode: currentMode,
    total: folder.files.length, ignored: folder.ignored,
    completed: 0, succeeded: 0, failed: 0, stopped: false,
    classes: Object.fromEntries(CLASS_NAMES[currentModel].map(name =>
      [name, { count: 0, bins: Array(10).fill(0) }]))
  };
  const startedAt = performance.now();
  let lastReportAt = startedAt;
  try {
    renderBatchReport(stats);
    updateBatchProgress(stats);
    await nextFrame();
    for (const file of folder.files) {
      if (batchCancelRequested) break;
      setStatus(`Processing ${stats.completed + 1}/${stats.total}: ${file.name}`);
      try {
        const { predictions } = await predictImage(file, activeSession, stats.model);
        recordPrediction(stats, predictions);
      } catch (error) {
        stats.failed++;
        $("batchErrorDetails").hidden = false;
        $("batchErrorSummary").textContent = `${stats.failed} images failed`;
        const row = document.createElement("li");
        row.textContent = `${file.webkitRelativePath || file.name}: ${error.message || String(error)}`;
        $("batchErrors").append(row);
      }
      stats.completed++;
      updateBatchProgress(stats, file);
      // Throttle chart rebuilds while allowing progress and Stop to stay responsive.
      if (performance.now() - lastReportAt >= 250) {
        renderBatchReport(stats);
        lastReportAt = performance.now();
      }
      await nextFrame();
    }
    stats.stopped = batchCancelRequested && stats.completed < stats.total;
    renderBatchReport(stats);
    updateBatchProgress(stats);
    setStatus(`Batch ${stats.stopped ? "stopped" : "completed"}: ${stats.completed}/${stats.total} images · ${((performance.now() - startedAt) / 1000).toFixed(1)} s`);
  } catch (error) {
    showError(error);
    setStatus("Batch processing failed");
  } finally {
    isRunning = false;
    isBatchRunning = false;
    updateControls();
  }
}

async function runSelectedInput() {
  if (selectedFolder) await runBatch();
  else if (selectedFile) await runInference(selectedFile);
}


// ============================================
// 13. FILE INPUT
// ============================================

selectButton.addEventListener(
  "click",
  () => {
    fileInput.click();
  }
);

fileInput.addEventListener(
  "change",
  async () => {
    const file = fileInput.files[0];

    if (!file) return;

    if (!IMAGE_TYPES.includes(file.type)) {
      showError(
        "Only JPG, PNG and WebP are supported."
      );

      fileInput.value = "";
      return;
    }

    selectedFile = file;
    selectedFolder = null;
    $("batchReport").hidden = true;
    $("batchProgressPanel").hidden = true;

    try {
      await runInference(file);

    } finally {
      // Allow choosing same image again.
      fileInput.value = "";
    }
  }
);


$("folderButton").addEventListener("click", () => $("folderInput").click());
$("folderInput").addEventListener("change", async () => {
  const allFiles = Array.from($("folderInput").files);
  $("folderInput").value = "";
  if (!allFiles.length || isRunning || isLoading) return;
  const files = allFiles.filter(file => IMAGE_TYPES.includes(file.type) ||
    (!file.type && /\.(jpe?g|png|webp)$/i.test(file.name)));
  if (!files.length) {
    showError("No supported images found. Select a folder containing JPG, PNG or WebP images.");
    return;
  }
  selectedFolder = {
    name: (allFiles[0].webkitRelativePath || "Selected folder").split("/")[0],
    files, ignored: allFiles.length - files.length
  };
  selectedFile = null;
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    preview.removeAttribute("src");
  }
  await runBatch();
});
$("batchRunButton").addEventListener("click", runBatch);
$("batchCancelButton").addEventListener("click", () => {
  batchCancelRequested = true;
  $("batchCancelButton").disabled = true;
  setStatus("Stopping after the current image...");
});


// ============================================
// 14. PROVIDER SELECTION
// ============================================

executionMode.addEventListener(
  "change",
  async () => {
    const selectedMode =
      executionMode.value;

    const loaded = await loadModel(modelChoice.value, selectedMode);
    if (loaded) await runSelectedInput();
  }
);


// ============================================
// 15. STARTUP
// ============================================

modelChoice.addEventListener("change", async () => {
  const loaded = await loadModel(modelChoice.value, executionMode.value);
  if (loaded) await runSelectedInput();
});

// Start with OpenNSFW2 on CPU by default.
loadModel("opennsfw2", "wasm");
