# co2-monitor

Dashboard Redesign: Dark Mode Enterprise Glassmorphism
Transform the Aira CO₂ Dashboard from its current Charcoal+Amber palette into a premium Midnight Glassmorphism style inspired by Claude and Midjourney's clean, ultra-premium UI paradigm. All content (text, labels, values, logic) remains unchanged.

User Review Required
IMPORTANT



WARNING

The .dash-card hover glow will transition from amber to a violet halo. Chart lines and gradients will use a Violet-to-Blue neon spectrum instead of amber.

Proposed Changes
Theme & Design Tokens
[MODIFY] 
theme.css
Complete rewrite of the .dark {} block and .dash-card styles:

Background: #0B0B0D (deeper midnight)
Card: rgba(255,255,255,0.03) with backdrop-filter: blur(12px)
Card border: 1px solid rgba(255,255,255,0.08)
Primary accent: Electric Violet #A855F7 / #BF5AF2
Secondary text: #949499
Status colors: Muted Ruby #FF375F for alerts, Sage Green #34d399 for stable
Chart palette: Violet → Pink → Blue spectrum
New tokens: --dash-violet, --dash-pink, --dash-blue for the neon triad
.dash-card: Glassmorphic with translucent bg, blur, subtle inner glow, 16px radius
.dash-card:hover: Violet border glow with box-shadow halo pulse
Root Layout
[MODIFY] 
Root.tsx
Update background gradient from amber tint to violet tint: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(168,85,247,0.04) 0%, #0B0B0D 70%)
Loading spinner: violet instead of amber
Add a subtle animated noise/grain overlay for premium depth feel
Sidebar
[MODIFY] 
Sidebar.tsx
Active nav item highlight: violet accent glow instead of amber
Icon active color: --dash-violet
Sidebar background: slightly elevated with ultra-subtle violet ambient border
Profile avatar ring: violet gradient hover effect
No text/label changes — only color and glow adjustments
Stat Cards
[MODIFY] 
StatCard.tsx
Top highlight bar: violet gradient instead of amber
Icon hover glow: violet
Gauge arc glow filter: updated to violet/blue spectrum
Needle glow: uses new palette
Health bar gradient: maintained green → amber → red (status semantic, unchanged)
No content changes — same titles, values, controls
Charts
[MODIFY] 
ChartSection.tsx
Area gradient fill: Violet-to-transparent
Line stroke: Electric Violet #A855F7
Neon glow filter: violet spectrum
Active dot: violet fill with violet glow
Tooltip: midnight bg with violet accent border
Legend dots: violet (actual) and muted grey (baseline) — label text unchanged
Prediction Card
[MODIFY] 
PredictionCard.tsx
Accent icon background: violet tint
+30min highlight row: violet accent background
Gauge arc: retains green → amber → red semantic gradient (status colors are functional, not decorative)
No label or value changes
Activity Feed
[MODIFY] 
ActivityFeed.tsx
Title bar accent: violet instead of red
activity.color for amber type: shift to violet
All alert text, descriptions, and times unchanged
Dashboard Page
[MODIFY] 
Dashboard.tsx
Status badge: violet accent for "OPTIMAL"
Training indicator: violet tint background
All content, grid layouts, and logic unchanged
Room Selector
[MODIFY] 
RoomSelector.tsx
Selected room highlight: violet text
Dropdown border accent: violet
Add room button: violet accent
All room names and functionality unchanged
Login Page
[MODIFY] 
LoginPage.tsx
Ambient glow orbs: violet and blue instead of amber and green
Grid overlay: violet tint
Card shadow: subtle violet halo
Logo gradient: violet spectrum
"CO₂ MONITOR" text color: violet
Divider accent: violet
Loading spinner: violet
All text content unchanged ("Welcome back", "Sign in to access...", etc.)
Settings Page
[MODIFY] 
Settings.tsx
Save button gradient: violet instead of amber
Active language/theme selection: violet accent
All options, labels, and descriptions unchanged
Modals
[MODIFY] 
NotificationsModal.tsx
[MODIFY] 
ProfileModal.tsx
Modal card: deeper midnight gradient with violet inner glow
Save/action buttons: violet gradient
Input focus rings: violet
All modal content, form fields, and notification text unchanged
