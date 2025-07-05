import { useState } from "react";

interface DropdownProps {
  options: string[];
  selected: string;
  onChange: (value: string) => void;
}

export default function Dropdown({ options, selected, onChange }: DropdownProps) {
  return (
    <select
      value={selected}
      onChange={(e) => onChange(e.target.value)}
      className="w-full p-2 rounded border border-gray-300 dark:border-gray-600 dark:bg-gray-800"
    >
      {options.map((opt) => (
        <option key={opt} value={opt}>
          {opt}
        </option>
      ))}
    </select>
  );
}
