const PALETTE = [
  "#5fd0c5", "#ffb877", "#ff8a9c", "#9b8cff",
  "#7ec96a", "#f5c84b", "#6fb3ff", "#e08bd8",
];

export function speakerColor(name: string | null | undefined): string {
  if (!name) return "#8a8278";
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return PALETTE[h % PALETTE.length];
}
