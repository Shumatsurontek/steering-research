interface Props {
  message: string;
  sub?: string;
}

export default function LoadingOverlay({ message, sub }: Props) {
  return (
    <div className="loading-overlay">
      <div className="loading-card">
        <div className="loading-spinner" />
        <div className="loading-text">{message}</div>
        {sub && <div className="loading-sub">{sub}</div>}
      </div>
    </div>
  );
}
