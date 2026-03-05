import React from 'react';

export default function PolicyComparator() {
    return (
        <div className="fade-in">
            <h1 className="gradient-text" style={{ fontSize: '2.5rem', marginBottom: '8px' }}>
                Policy Comparator
            </h1>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '32px' }}>
                Side-by-side policy comparison matrix (Feature 01 / Phase 2 - Pending)
            </p>
            <div style={{ background: 'var(--bg-secondary)', padding: '24px', borderRadius: '12px', border: '1px solid var(--glass-border)' }}>
                <p style={{ color: 'var(--text-muted)' }}>Comparison Table will be built here.</p>
            </div>
        </div>
    );
}
