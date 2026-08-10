import { expect, test } from "@playwright/test";

import {
  extractSemanticOccupancyComponents,
  SEMANTIC_OCCUPANCY_CLASS_NAMES,
  type SemanticOccupancyArtifact,
} from "../src/lib/semantic-occupancy";

const HEIGHT = 12;
const WIDTH = 12;
const CLASS_COUNT = SEMANTIC_OCCUPANCY_CLASS_NAMES.length;

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
  expect(components[0].meanConfidence).toBeCloseTo(230 / 255, 6);
  expect(components[1].principalAxisRadians).toBeCloseTo(0, 6);
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
