import {
  expect,
  test,
  type Locator,
  type Page,
} from "@playwright/test";

import {
  extractSemanticOccupancyComponents,
  SEMANTIC_OCCUPANCY_CLASS_NAMES,
  type SemanticOccupancyArtifact,
} from "../src/lib/semantic-occupancy";

const HEIGHT = 12;
const WIDTH = 12;
const CLASS_COUNT = SEMANTIC_OCCUPANCY_CLASS_NAMES.length;
const DEMO_PATH = "/semantic-occupancy-demo";
const HDRI_PATH =
  "/assets/semantic-occupancy/poly-haven/studio_small_09_1k.hdr";

async function webGLPaintState(canvas: Locator) {
  return canvas.evaluate((element) => {
    const target = element as HTMLCanvasElement;
    const gl =
      target.getContext("webgl2") ?? target.getContext("webgl");
    if (!gl) {
      return {
        colorfulSamples: 0,
        hash: 0,
        height: 0,
        paintedSamples: 0,
        variance: 0,
        width: 0,
      };
    }
    gl.finish();
    const width = gl.drawingBufferWidth;
    const height = gl.drawingBufferHeight;
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(
      0,
      0,
      width,
      height,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      pixels,
    );
    const background = [pixels[0], pixels[1], pixels[2]];
    let colorfulSamples = 0;
    let hash = 2166136261;
    let luminanceSquaredSum = 0;
    let luminanceSum = 0;
    let paintedSamples = 0;
    let samples = 0;
    for (let offset = 0; offset < pixels.length; offset += 16) {
      const red = pixels[offset];
      const green = pixels[offset + 1];
      const blue = pixels[offset + 2];
      const difference =
        Math.abs(red - background[0]) +
        Math.abs(green - background[1]) +
        Math.abs(blue - background[2]);
      if (difference > 18) paintedSamples++;
      if (Math.max(red, green, blue) - Math.min(red, green, blue) > 12) {
        colorfulSamples++;
      }
      const luminance = (red + green + blue) / 3;
      luminanceSum += luminance;
      luminanceSquaredSum += luminance * luminance;
      hash = Math.imul(hash ^ red, 16777619);
      hash = Math.imul(hash ^ green, 16777619);
      hash = Math.imul(hash ^ blue, 16777619);
      samples++;
    }
    const mean = luminanceSum / samples;
    return {
      colorfulSamples,
      hash: hash >>> 0,
      height,
      paintedSamples,
      variance: luminanceSquaredSum / samples - mean * mean,
      width,
    };
  });
}

async function openDemo(
  page: Page,
  viewport: { width: number; height: number },
) {
  const errors: string[] = [];
  const responseStatuses = new Map<string, number>();
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(`console: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => {
    errors.push(`page: ${error.message}`);
  });
  page.on("response", (response) => {
    const url = new URL(response.url());
    responseStatuses.set(url.pathname, response.status());
  });
  await page.setViewportSize(viewport);
  await page.goto(DEMO_PATH, { waitUntil: "domcontentloaded" });
  const region = page.getByRole("region", {
    name: "3D semantic occupancy",
  });
  const canvas = region.locator("canvas").first();
  await expect(canvas).toBeVisible({ timeout: 20_000 });
  await expect(canvas).toHaveAttribute("data-engine", /three\.js/, {
    timeout: 20_000,
  });
  await expect(canvas).toHaveAttribute(
    "aria-label",
    "Interactive 3D semantic occupancy scene",
  );
  await expect
    .poll(async () => (await webGLPaintState(canvas)).paintedSamples, {
      timeout: 20_000,
    })
    .toBeGreaterThan(1_000);
  await expect
    .poll(() => responseStatuses.get(HDRI_PATH), {
      message: "The CC0 HDRI should load from the local app",
    })
    .toBe(200);
  return { canvas, errors, region };
}

function artifact(): SemanticOccupancyArtifact {
  const cellCount = CLASS_COUNT * HEIGHT * WIDTH;
  return {
    formatVersion: 1,
    flags: 1,
    sampleCount: 1,
    classCount: CLASS_COUNT,
    height: HEIGHT,
    width: WIDTH,
    directory: [{ hashHigh: 0, hashLow: 1, row: 0 }],
    probability: new Uint8Array(cellCount),
    teacher: new Uint8Array(cellCount),
    validBits: new Uint8Array(Math.ceil(cellCount / 8)).fill(255),
  };
}

function paint(
  target: Uint8Array,
  classIndex: number,
  cells: readonly [number, number][],
  value: number,
) {
  for (const [row, col] of cells) {
    const index = (classIndex * HEIGHT + row) * WIDTH + col;
    target[index] = value;
  }
}

test("extracts connected components and estimates their principal axes", () => {
  const semantic = artifact();
  const horizontal = Array.from(
    { length: 6 },
    (_, index) => [3, index + 2] as [number, number],
  );
  const vertical = Array.from(
    { length: 5 },
    (_, index) => [index + 6, 9] as [number, number],
  );
  paint(semantic.probability, 5, horizontal, 230);
  paint(semantic.probability, 5, vertical, 204);

  const components = extractSemanticOccupancyComponents({
    artifact: semantic,
    row: 0,
    classIndices: [5],
    mode: "prediction",
    threshold: 0.5,
  });

  expect(components).toHaveLength(2);
  expect(components[0]).toMatchObject({
    className: "vehicle",
    cellCount: 6,
    centroidRow: 3,
    minCol: 2,
    maxCol: 7,
  });
  expect(Math.abs(components[0].principalAxisRadians)).toBeCloseTo(
    Math.PI / 2,
    6,
  );
  expect(components[0].majorSpanCells).toBeCloseTo(6, 6);
  expect(components[0].minorSpanCells).toBeCloseTo(1, 6);
  expect(components[0].meanConfidence).toBeCloseTo(230 / 255, 6);
  expect(components[1].principalAxisRadians).toBeCloseTo(0, 6);
  expect(components[1].majorSpanCells).toBeCloseTo(5, 6);
  expect(components[1].minorSpanCells).toBeCloseTo(1, 6);
});

test("keeps component footprints faithful for single and diagonal cells", () => {
  const semantic = artifact();
  paint(semantic.probability, 5, [[1, 1]], 255);
  paint(
    semantic.probability,
    6,
    [
      [4, 4],
      [5, 5],
      [6, 6],
      [7, 7],
    ],
    255,
  );

  const components = extractSemanticOccupancyComponents({
    artifact: semantic,
    row: 0,
    classIndices: [5, 6],
    mode: "prediction",
    threshold: 0.5,
  });
  const diagonal = components.find(
    (component) => component.className === "vulnerable_road_user",
  );
  const single = components.find(
    (component) => component.className === "vehicle",
  );

  expect(single?.majorSpanCells).toBeCloseTo(1, 6);
  expect(single?.minorSpanCells).toBeCloseTo(1, 6);
  expect(diagonal?.principalAxisRadians).toBeCloseTo(Math.PI / 4, 6);
  expect(diagonal?.majorSpanCells).toBeCloseTo(4 * Math.SQRT2, 6);
  expect(diagonal?.minorSpanCells).toBeCloseTo(Math.SQRT2, 6);
  expect(diagonal?.minorSpanCells).toBeLessThan(4);
});

test("keeps false positives and false negatives as separate error objects", () => {
  const semantic = artifact();
  const falsePositive: [number, number][] = [
    [4, 4],
    [4, 5],
  ];
  const falseNegative: [number, number][] = [
    [5, 5],
    [5, 6],
  ];
  paint(semantic.probability, 6, falsePositive, 242);
  paint(semantic.teacher!, 6, falseNegative, 255);

  const components = extractSemanticOccupancyComponents({
    artifact: semantic,
    row: 0,
    classIndices: [6],
    mode: "error",
    threshold: 0.5,
  });

  expect(components).toHaveLength(2);
  expect(components.map((component) => component.errorKind).sort()).toEqual([
    "fn",
    "fp",
  ]);
  expect(components.every((component) => component.cellCount === 2)).toBe(
    true,
  );
});

test("honors teacher validity, minimum size, and object count bounds", () => {
  const semantic = artifact();
  paint(
    semantic.teacher!,
    7,
    [
      [1, 1],
      [1, 2],
      [5, 5],
      [9, 9],
    ],
    255,
  );
  const invalidIndex = (7 * HEIGHT + 1) * WIDTH + 1;
  semantic.validBits![invalidIndex >> 3] &=
    ~(1 << (invalidIndex & 7));

  const components = extractSemanticOccupancyComponents({
    artifact: semantic,
    row: 0,
    classIndices: [7],
    mode: "teacher",
    threshold: 0.5,
    minCells: 1,
    maxComponents: 2,
  });

  expect(components).toHaveLength(2);
  expect(components.every((component) => component.cellCount === 1)).toBe(
    true,
  );
  expect(
    extractSemanticOccupancyComponents({
      artifact: semantic,
      row: 0,
      classIndices: [7],
      mode: "teacher",
      threshold: 0.5,
      minCells: 2,
    }),
  ).toEqual([]);
});

test("renders the interactive premium scene and all semantic modes", async ({
  page,
}, testInfo) => {
  testInfo.setTimeout(90_000);
  const { canvas, errors, region } = await openDemo(page, {
    width: 1440,
    height: 1000,
  });
  const orbitPaint = await webGLPaintState(canvas);
  expect(orbitPaint.width).toBeGreaterThan(700);
  expect(orbitPaint.height).toBeGreaterThan(500);
  expect(orbitPaint.colorfulSamples).toBeGreaterThan(500);
  expect(orbitPaint.variance).toBeGreaterThan(20);

  await region.getByRole("button", { name: "Top view" }).click();
  await page.waitForTimeout(850);
  const topPaint = await webGLPaintState(canvas);
  expect(topPaint.hash).not.toBe(orbitPaint.hash);

  await region.getByRole("button", { name: "Ego view" }).click();
  await page.waitForTimeout(850);
  const egoPaint = await webGLPaintState(canvas);
  expect(egoPaint.hash).not.toBe(topPaint.hash);
  expect(egoPaint.paintedSamples).toBeGreaterThan(500);

  await region.getByRole("button", { name: "Top view" }).click();
  await page.waitForTimeout(850);
  const predictionPaint = await webGLPaintState(canvas);
  await region.getByRole("tab", { name: "Teacher" }).click();
  await page.waitForTimeout(250);
  const teacherPaint = await webGLPaintState(canvas);
  expect(teacherPaint.hash).not.toBe(predictionPaint.hash);

  await region.getByRole("tab", { name: "Error" }).click();
  await page.waitForTimeout(250);
  await expect(region).toContainText("FP");
  await expect(region).toContainText("FN");
  const errorPaint = await webGLPaintState(canvas);
  expect(errorPaint.hash).not.toBe(teacherPaint.hash);
  expect(errorPaint.colorfulSamples).toBeGreaterThan(250);

  const layout = await region.evaluate((element) => {
    const scene = element.querySelector("canvas")?.parentElement;
    const camera = element.querySelector(
      '[role="group"][aria-label="Semantic occupancy camera"]',
    );
    if (!scene || !camera) return null;
    const sceneRect = scene.getBoundingClientRect();
    const cameraRect = camera.getBoundingClientRect();
    return {
      cameraInside:
        cameraRect.left >= sceneRect.left &&
        cameraRect.right <= sceneRect.right &&
        cameraRect.top >= sceneRect.top &&
        cameraRect.bottom <= sceneRect.bottom,
      horizontalOverflow:
        document.documentElement.scrollWidth -
        document.documentElement.clientWidth,
    };
  });
  expect(layout).toEqual({ cameraInside: true, horizontalOverflow: 0 });
  await page.screenshot({
    fullPage: true,
    path: testInfo.outputPath("semantic-occupancy-desktop.png"),
  });
  expect(errors).toEqual([]);
});

test("keeps the WebGL scene usable on a mobile viewport", async ({
  page,
}, testInfo) => {
  testInfo.setTimeout(90_000);
  const { canvas, errors, region } = await openDemo(page, {
    width: 390,
    height: 844,
  });
  const orbitPaint = await webGLPaintState(canvas);
  expect(orbitPaint.width).toBeGreaterThan(300);
  expect(orbitPaint.height).toBeGreaterThan(350);
  expect(orbitPaint.colorfulSamples).toBeGreaterThan(200);
  expect(orbitPaint.variance).toBeGreaterThan(12);

  await region.getByRole("button", { name: "Ego view" }).click();
  await page.waitForTimeout(850);
  const egoPaint = await webGLPaintState(canvas);
  expect(egoPaint.hash).not.toBe(orbitPaint.hash);

  await region.getByRole("tab", { name: "Teacher" }).click();
  await page.waitForTimeout(250);
  await expect(
    region.getByRole("tab", { name: "Teacher" }),
  ).toHaveAttribute("aria-selected", "true");

  const layout = await page.evaluate(() => {
    const region = document.querySelector(
      '[aria-label="3D semantic occupancy"]',
    );
    const canvas = region?.querySelector("canvas");
    const camera = region?.querySelector(
      '[role="group"][aria-label="Semantic occupancy camera"]',
    );
    if (!region || !canvas || !camera) return null;
    const regionRect = region.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    const cameraRect = camera.getBoundingClientRect();
    return {
      cameraInside:
        cameraRect.left >= canvasRect.left &&
        cameraRect.right <= canvasRect.right &&
        cameraRect.top >= canvasRect.top &&
        cameraRect.bottom <= canvasRect.bottom,
      regionInside:
        regionRect.left >= 0 &&
        regionRect.right <= document.documentElement.clientWidth,
      horizontalOverflow:
        document.documentElement.scrollWidth -
        document.documentElement.clientWidth,
    };
  });
  expect(layout).toEqual({
    cameraInside: true,
    horizontalOverflow: 0,
    regionInside: true,
  });
  await page.screenshot({
    fullPage: true,
    path: testInfo.outputPath("semantic-occupancy-mobile.png"),
  });
  expect(errors).toEqual([]);
});
