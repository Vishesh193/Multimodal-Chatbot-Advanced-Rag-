import React, { useState, useEffect } from 'react';
import { Briefcase, GitCompare, Loader2, AlertCircle } from 'lucide-react';
import { api } from '../api/client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function PolicyComparator() {
    const [policies, setPolicies] = useState([]);
    const [policyA, setPolicyA] = useState('');
    const [policyB, setPolicyB] = useState('');
    const [query, setQuery] = useState('');
    const [language, setLanguage] = useState('english');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    // Fetch ingested policies on mount
    useEffect(() => {
        api.getPolicies('health')
            .then(res => setPolicies(res.policies))
            .catch(err => console.error("Failed to load policies:", err));
    }, []);

    const handleCompare = async (e) => {
        e.preventDefault();
        if (!policyA || !policyB || !query) {
            setError('Please select two policies and enter a query.');
            return;
        }
        if (policyA === policyB) {
            setError('Please select two DIFFERENT policies to compare.');
            return;
        }

        setLoading(true);
        setError('');
        setResult(null);

        try {
            const response = await api.comparePolicies({
                policy_name_a: policyA,
                policy_name_b: policyB,
                query: query,
                language: language
            });
            setResult(response.comparison_table);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Comparison failed.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fade-in" style={{ maxWidth: '1000px', margin: '0 auto' }}>
            <h1 className="gradient-text" style={{ fontSize: '2.5rem', marginBottom: '8px' }}>
                Policy Comparator
            </h1>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '32px' }}>
                Compare two uploaded policies side-by-side with high precision.
            </p>

            <form onSubmit={handleCompare} className="glass-panel" style={{ padding: '32px', marginBottom: '32px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'minmax(200px, 1fr) auto minmax(200px, 1fr)', gap: '24px', alignItems: 'flex-end', marginBottom: '24px' }}>

                    <div>
                        <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Policy A</label>
                        <select
                            value={policyA}
                            onChange={(e) => setPolicyA(e.target.value)}
                            style={{ width: '100%', padding: '16px', borderRadius: '12px', border: '1px solid var(--glass-border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '1rem' }}
                        >
                            <option value="">Select Policy</option>
                            {policies.map(p => (
                                <option key={p.policy_id} value={p.policy_name}>{p.policy_name}</option>
                            ))}
                        </select>
                    </div>

                    <div style={{ paddingBottom: '16px', color: 'var(--text-muted)' }}>
                        <GitCompare size={32} />
                    </div>

                    <div>
                        <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Policy B</label>
                        <select
                            value={policyB}
                            onChange={(e) => setPolicyB(e.target.value)}
                            style={{ width: '100%', padding: '16px', borderRadius: '12px', border: '1px solid var(--glass-border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '1rem' }}
                        >
                            <option value="">Select Policy</option>
                            {policies.map(p => (
                                <option key={p.policy_id} value={p.policy_name}>{p.policy_name}</option>
                            ))}
                        </select>
                    </div>
                </div>

                <div style={{ marginBottom: '24px' }}>
                    <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Aspect to Compare</label>
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="e.g. Do they cover maternity expenses and what is the waiting period?"
                        style={{ width: '100%', padding: '16px', borderRadius: '12px', border: '1px solid var(--glass-border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '1rem' }}
                    />
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Language:</span>
                        <select
                            value={language}
                            onChange={(e) => setLanguage(e.target.value)}
                            style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid var(--glass-border)', background: 'var(--bg-color)', color: 'var(--text-primary)' }}
                        >
                            <option value="english">English</option>
                            <option value="hindi">Hindi</option>
                            <option value="tamil">Tamil</option>
                        </select>
                    </div>

                    <button
                        type="submit"
                        className="btn-primary"
                        disabled={loading}
                        style={{ padding: '12px 32px', opacity: loading ? 0.7 : 1 }}
                    >
                        {loading ? <><Loader2 className="animate-spin" size={18} /> Analyzing Policies...</> : 'Compare Policies'}
                    </button>
                </div>

                {error && (
                    <div style={{ marginTop: '24px', padding: '16px', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', color: 'var(--error)', display: 'flex', gap: '8px' }}>
                        <AlertCircle size={20} /> {error}
                    </div>
                )}
            </form>

            {/* Results Container with Markdown Support */}
            {result && (
                <div className="glass-panel fade-in" style={{ padding: '32px', background: 'var(--bg-color)', overflowX: 'auto' }}>
                    <style dangerouslySetInnerHTML={{
                        __html: `
            .markdown-body table { width: 100%; border-collapse: collapse; margin-bottom: 1rem; }
            .markdown-body th, .markdown-body td { padding: 12px; border: 1px solid var(--glass-border); text-align: left; }
            .markdown-body th { background: var(--bg-secondary); }
            .markdown-body p { margin-bottom: 1rem; }
            .markdown-body h1, .markdown-body h2, .markdown-body h3 { margin-top: 1.5rem; margin-bottom: 1rem; color: var(--accent-blue); }
            .markdown-body ul, .markdown-body ol { margin-bottom: 1rem; padding-left: 2rem; }
            .markdown-body ul { list-style-type: disc; }
          `}} />
                    <div className="markdown-body" style={{
                        color: 'var(--text-primary)',
                        lineHeight: 1.6
                    }}>
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {result}
                        </ReactMarkdown>
                    </div>
                </div>
            )}
        </div>
    );
}
