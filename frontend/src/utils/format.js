export const cls = (...parts) => parts.filter(Boolean).join(" ");

export const fmt = (value, digits = 2) => Number(value).toFixed(digits);

export const fmtGb = (bytes, digits = 1) => (Number(bytes) / 1024 ** 3).toFixed(digits);
