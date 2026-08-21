"use client";

import { useMemo } from "react";

import { SemanticOccupancyView } from "@/components/player/semantic-occupancy-view";
import {
  SEMANTIC_OCCUPANCY_CLASS_NAMES,
  type SemanticOccupancyArtifact,
} from "@/lib/semantic-occupancy";

const HEIGHT = 450;
const WIDTH = 300;
const CLASS_COUNT = SEMANTIC_OCCUPANCY_CLASS_NAMES.length;

type RasterTarget = "both" | "prediction" | "teacher";

function createMockArtifact(): SemanticOccupancyArtifact {
  const cellCount = CLASS_COUNT * HEIGHT * WIDTH;
  const probability = new Uint8Array(cellCount);
  const teacher = new Uint8Array(cellCount);

  const paint = (
    classIndex: number,
    minRow: number,
    maxRow: number,
    minCol: number,
    maxCol: number,
    value: number,
    target: RasterTarget = "both",
  ) => {
    for (let row = minRow; row <= maxRow; row++) {
      for (let col = minCol; col <= maxCol; col++) {
        const index = (classIndex * HEIGHT + row) * WIDTH + col;
        if (target !== "teacher") probability[index] = value;
        if (target !== "prediction") teacher[index] = value;
      }
    }
  };

  paint(0, 20, 440, 52, 247, 168);
  paint(2, 165, 220, 52, 247, 188);
  for (let row = 30; row < 430; row += 28) {
    paint(1, row, Math.min(row + 12, 449), 123, 126, 226);
    paint(1, row, Math.min(row + 12, 449), 173, 176, 226);
  }
  for (let row = 184; row <= 210; row += 7) {
    paint(3, row, row + 3, 72, 227, 238);
  }
  paint(4, 227, 230, 62, 237, 244);

  paint(5, 258, 276, 145, 151, 238, "prediction");
  paint(5, 260, 278, 144, 150, 255, "teacher");
  paint(5, 238, 246, 183, 200, 224, "prediction");
  paint(5, 236, 244, 180, 197, 255, "teacher");
  paint(5, 178, 198, 103, 111, 218, "prediction");
  paint(5, 178, 196, 105, 113, 255, "teacher");
  paint(5, 220, 235, 225, 232, 210, "prediction");
  paint(5, 142, 158, 160, 166, 255, "teacher");

  paint(6, 270, 274, 121, 124, 232, "prediction");
  paint(6, 271, 275, 122, 125, 255, "teacher");
  paint(6, 220, 225, 209, 212, 216, "prediction");
  paint(6, 219, 224, 207, 210, 255, "teacher");
  paint(6, 315, 319, 188, 191, 255, "teacher");

  paint(7, 202, 211, 72, 84, 225, "prediction");
  paint(7, 203, 212, 73, 85, 255, "teacher");
  paint(7, 305, 307, 213, 215, 230);
  paint(7, 248, 250, 220, 222, 224);
  paint(7, 280, 283, 62, 65, 218);
  paint(7, 245, 249, 70, 73, 232);
  paint(7, 260, 265, 90, 92, 228);
  paint(7, 128, 134, 216, 233, 215, "prediction");
  paint(7, 130, 136, 214, 231, 255, "teacher");

  return {
    formatVersion: 1,
    flags: 1,
    sampleCount: 1,
    classCount: CLASS_COUNT,
    height: HEIGHT,
    width: WIDTH,
    directory: [{ hashHigh: 0, hashLow: 1, row: 0 }],
    probability,
    teacher,
    validBits: new Uint8Array(Math.ceil(cellCount / 8)).fill(255),
  };
}

export default function SemanticOccupancyDemoPage() {
  const artifact = useMemo(createMockArtifact, []);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="font-mono text-[10px] uppercase text-cyan-400">
            Local fixture
          </p>
          <h2 className="text-lg font-semibold text-slate-100">
            Semantic occupancy mock scene
          </h2>
        </div>
        <span className="rounded border border-slate-700 bg-slate-950/60 px-2 py-1 font-mono text-[9px] uppercase text-slate-400">
          No model or dataset required
        </span>
      </div>
      <SemanticOccupancyView
        artifact={artifact}
        demoEnvironment
        row={0}
        status="ready"
      />
    </div>
  );
}
