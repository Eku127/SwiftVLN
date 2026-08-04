interface SettingsIconProps {
  size?: number;
}

export function SettingsIcon({ size = 18 }: SettingsIconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <path
        d="M12 15.2a3.2 3.2 0 1 0 0-6.4 3.2 3.2 0 0 0 0 6.4Z"
        stroke="currentColor"
        strokeWidth="1.6"
      />
      <path
        d="M19.4 13.5a7.4 7.4 0 0 0 .1-3l1.7-1a.8.8 0 0 0 .3-1.1l-1.6-2.8a.8.8 0 0 0-1-.35l-2 .6a7.5 7.5 0 0 0-2.6-1.5l-.3-2.1a.8.8 0 0 0-.8-.7h-3.2a.8.8 0 0 0-.8.7l-.3 2.1a7.5 7.5 0 0 0-2.6 1.5l-2-.6a.8.8 0 0 0-1 .35L2.5 8.4a.8.8 0 0 0 .3 1.1l1.7 1a7.4 7.4 0 0 0 0 3l-1.7 1a.8.8 0 0 0-.3 1.1l1.6 2.8a.8.8 0 0 0 1 .35l2-.6c.8.6 1.7 1.1 2.6 1.5l.3 2.1a.8.8 0 0 0 .8.7h3.2a.8.8 0 0 0 .8-.7l.3-2.1c.9-.4 1.8-.9 2.6-1.5l2 .6a.8.8 0 0 0 1-.35l1.6-2.8a.8.8 0 0 0-.3-1.1l-1.7-1Z"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinejoin="round"
      />
    </svg>
  );
}
