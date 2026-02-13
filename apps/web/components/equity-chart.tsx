"use client";

import { useEffect, useRef } from "react";
import { ColorType, createChart } from "lightweight-charts";

type Point = {
  time: string;
  value: number;
};

type SeriesInput = {
  id: string;
  points: Point[];
  color?: string;
};

export function EquityChart({
  points = [],
  series = [],
}: {
  points?: Point[];
  series?: SeriesInput[];
}) {
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!ref.current) {
      return;
    }

    const chart = createChart(ref.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#8795aa",
      },
      grid: {
        vertLines: { color: "rgba(120, 140, 170, 0.16)" },
        horzLines: { color: "rgba(120, 140, 170, 0.16)" },
      },
      height: 260,
      rightPriceScale: {
        borderVisible: false,
      },
      timeScale: {
        borderVisible: false,
      },
      crosshair: {
        vertLine: { color: "rgba(0, 122, 255, 0.35)", labelVisible: false },
        horzLine: { color: "rgba(0, 122, 255, 0.35)", labelVisible: false },
      },
    });

    if (series.length > 1) {
      const palette = ["#007aff", "#34c759", "#ff9500"];
      series.slice(0, 3).forEach((item, index) => {
        const line = chart.addLineSeries({
          color: item.color ?? palette[index] ?? "#007aff",
          lineWidth: 2,
        });
        line.setData(item.points.map((p) => ({ time: p.time as never, value: p.value })));
      });
    } else {
      const area = chart.addAreaSeries({
        lineColor: "#007aff",
        topColor: "rgba(0, 122, 255, 0.28)",
        bottomColor: "rgba(0, 122, 255, 0.02)",
        lineWidth: 2,
      });
      const source = series.length === 1 ? (series[0]?.points ?? []) : points;
      area.setData(source.map((p) => ({ time: p.time as never, value: p.value })));
    }

    chart.timeScale().fitContent();

    const resize = () => {
      if (!ref.current) {
        return;
      }
      chart.applyOptions({ width: ref.current.clientWidth });
    };

    resize();
    window.addEventListener("resize", resize);

    return () => {
      window.removeEventListener("resize", resize);
      chart.remove();
    };
  }, [points, series]);

  return <div ref={ref} className="w-full" aria-label="Equity chart" />;
}
