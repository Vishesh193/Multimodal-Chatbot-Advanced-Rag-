import React, { useState, useEffect } from 'react';
import { FileText, Wand2, Loader2, AlertCircle } from 'lucide-react';
import { api } from '../api/client';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

export default function ClaimChecklist() {
    const [policies, setPolicies] = useState([]);
    const [selectedPolicy, setSelectedPolicy] = useState('');
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

    const handleGenerate = async (e) => {
        e.preventDefault();
        if (!query) {
            setError('Please describe your claim issue.');
            return;
        }

        setLoading(true);
        setError('');
        setResult(null);

        try {
            const response = await api.getClaimChecklist({
                query: query,
                policy_name: selectedPolicy || undefined,
                language: language
            });
            setResult(response.checklist);
        } catch (err) {
            setError(err.response?.data?.detail || err.message || 'Checklist generation failed.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fade-in" style={{ maxWidth: '900px', margin: '0 auto' }}>
            <h1 className="gradient-text" style={{ fontSize: '2.5rem', marginBottom: '8px' }}>
                Claim Action Plan Tracker
            </h1>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '32px' }}>
                Get a personalized, step-by-step checklist of documents and actions needed for your specific hospital claim.
            </p>

            <form onSubmit={handleGenerate} className="glass-panel" style={{ padding: '32px', marginBottom: '32px' }}>
                <div style={{ marginBottom: '24px' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', fontSize: '1rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                        <FileText size={18} color="var(--accent-purple)" /> Describe your situation
                    </label>
                    <textarea
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="e.g. I was admitted for dengue fever for 3 days and need to file a reimbursement claim."
                        rows="3"
                        style={{
                            width: '100%', padding: '16px', borderRadius: '12px', border: '1px solid var(--glass-border)',
                            background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: '1rem', resize: 'vertical'
                        }}
                    />
                </div>

                <div style={{ display: 'flex', gap: '24px', marginBottom: '32px' }}>
                    <div style={{ flex: 1 }}>
                        <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Select Policy (Optional)</label>
                        <select
                            value={selectedPolicy}
                            onChange={(e) => setSelectedPolicy(e.target.value)}
                            style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--glass-border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
                        >
                            <option value="">General Claim Guidance</option>
                            {policies.map(p => (
                                <option key={p.policy_id} value={p.policy_name}>{p.policy_name}</option>
                            ))}
                        </select>
                        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '8px' }}>If selected, the checklist will be specific to your exact policy terms.</p>
                    </div>

                    <div style={{ flex: 1 }}>
                        <label style={{ display: 'block', marginBottom: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Language</label>
                        <select
                            value={language}
                            onChange={(e) => setLanguage(e.target.value)}
                            style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid var(--glass-border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)' }}
                        >
                            <option value="english">English</option>
                            <option value="hindi">Hindi</option>
                            <option value="tamil">Tamil</option>
                            <option value="telugu">Telugu</option>
                        </select>
                    </div>
                </div>

                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <button
                        type="submit"
                        className="btn-primary"
                        disabled={loading}
                        style={{ padding: '12px 32px', opacity: loading ? 0.7 : 1 }}
                    >
                        {loading ? <><Loader2 className="animate-spin" size={18} /> Generating Checklist...</> : <><Wand2 size={18} /> Generate Action Plan</>}
                    </button>
                </div>

                {error && (
                    <div style={{ marginTop: '24px', padding: '16px', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', color: 'var(--error)', display: 'flex', gap: '8px' }}>
                        <AlertCircle size={20} /> {error}
                    </div>
                )}
            </form>

            {/* Markdown Results Container */}
            {result && (
                <div className="glass-panel fade-in" style={{ padding: '32px', background: 'var(--bg-color)' }}>
                    <style dangerouslySetInnerHTML={{
                        __html: `
            .claim-markdown table { width: 100%; border-collapse: collapse; margin: 1.5rem 0; }
            .claim-markdown th, .claim-markdown td { padding: 12px; border: 1px solid var(--glass-border); text-align: left; }
            .claim-markdown th { background: var(--bg-secondary); color: var(--accent-purple); }
            .claim-markdown h1, .claim-markdown h2, .claim-markdown h3 { margin-top: 1.5rem; margin-bottom: 1rem; color: var(--accent-blue); display: flex; align-items: center; gap: 8px;}
            .claim-markdown ul, .claim-markdown ol { margin-bottom: 1.5rem; padding-left: 2rem; }
            .claim-markdown li { margin-bottom: 0.5rem; }
            .claim-markdown strong { color: var(--text-primary); }
            .claim-markdown hr { border: none; border-top: 1px solid var(--glass-border); margin: 2rem 0; }
            .claim-markdown blockquote { border-left: 4px solid var(--warning); padding-left: 1rem; color: var(--text-secondary); background: rgba(245, 158, 11, 0.1); padding: 1rem; border-radius: 0 8px 8px 0; }
          `}} />
                    <div className="claim-markdown" style={{
                        color: 'var(--text-primary)',
                        lineHeight: 1.7,
                        fontSize: '1.05rem'
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
