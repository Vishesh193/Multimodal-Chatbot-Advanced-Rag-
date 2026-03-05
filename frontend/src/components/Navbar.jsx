import React, { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Shield } from 'lucide-react';

export default function Navbar() {
    const location = useLocation();
    const navigate = useNavigate();
    const [scrolled, setScrolled] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 20);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const scrollTo = (id) => {
        if (location.pathname !== '/') {
            navigate('/#' + id);
        } else {
            setTimeout(() => {
                document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
            }, 100);
        }
    };

    return (
        <nav style={{
            height: 'var(--navbar-height)',
            background: 'white',
            borderBottom: scrolled ? '1px solid var(--glass-border)' : '1px solid transparent',
            position: 'sticky',
            top: 0,
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 5%',
            boxShadow: scrolled ? '0 4px 6px -1px rgba(0, 0, 0, 0.05)' : 'none',
            transition: 'all 0.3s ease'
        }}>
            <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: '8px', textDecoration: 'none' }}>
                <Shield size={28} color="var(--accent-deep)" />
                <span style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-deep)', fontFamily: 'Poppins, sans-serif' }}>
                    InsureAI
                </span>
            </Link>

            <div style={{ display: 'flex', gap: '32px', alignItems: 'center' }}>
                <button onClick={() => scrollTo('home')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.95rem' }}>Home</button>
                <button onClick={() => scrollTo('features')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.95rem' }}>Features</button>
                <Link to="/app/chat" style={{ textDecoration: 'none', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.95rem' }}>Insurance Assistant</Link>
                <Link to="/app/comparator" style={{ textDecoration: 'none', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.95rem' }}>Compare Policies</Link>
                <Link to="/app/checklist" style={{ textDecoration: 'none', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.95rem' }}>Claim Guide</Link>
                <button onClick={() => scrollTo('faq')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.95rem' }}>FAQ</button>
                <button onClick={() => scrollTo('contact')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.95rem' }}>Contact</button>
            </div>

            <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
                <Link to="/app/overview" style={{ textDecoration: 'none', color: 'var(--text-primary)', fontWeight: 600, fontSize: '0.95rem' }}>Login</Link>
                <Link to="/app/chat" className="btn-primary" style={{ textDecoration: 'none' }}>Try Assistant</Link>
            </div>
        </nav>
    );
}
