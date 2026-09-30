-- ============================================================
-- CS2 Player Builder leaderboard
-- ============================================================
--
-- Run this once in the Supabase dashboard: SQL Editor -> New
-- query -> paste -> Run.
--
-- Anyone can read the leaderboard and add a score. Nobody can
-- change or delete scores from the website (only you, from the
-- dashboard).

create table if not exists public.scores (
    id          bigint generated always as identity primary key,
    name        text        not null,
    score       numeric(5, 1) not null,
    grade       text        not null,
    build       jsonb       not null,
    created_at  timestamptz not null default now(),

    constraint name_length  check (char_length(btrim(name)) between 1 and 20),
    constraint score_range  check (score between 0 and 100),
    constraint grade_value  check (grade in ('S', 'A', 'B', 'C', 'D')),
    constraint build_object check (jsonb_typeof(build) = 'object')
);

create index if not exists scores_score_idx on public.scores (score desc, created_at);

-- Row level security: the website's public key can only read
-- and insert, never update or delete.
alter table public.scores enable row level security;

drop policy if exists "Anyone can read scores" on public.scores;
create policy "Anyone can read scores"
    on public.scores for select
    to anon
    using (true);

drop policy if exists "Anyone can add a score" on public.scores;
create policy "Anyone can add a score"
    on public.scores for insert
    to anon
    with check (true);
