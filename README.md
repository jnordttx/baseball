# MLB Strategy Engine: 2026 Deployment Analytics

[Streamlit App](https://baseballgm.streamlit.app/) 

## Project Overview
I built a front office dashboard that could be used by MLB teams to identify what players are over/under-performing their contracts based on publicly available WAR & Salary $AAV data.  

### The Core Logic: $8.5M/WAR Market Theory
First, it's good to define what 'WAR' is. In baseball, WAR stands for Wins Above Replacement and it's one of the metrics used to determine a player's value relative to a replacement level player. 
The higher the WAR, the better the player. 
The engine is built on the economic principle that **1.0 Wins Above Replacement (WAR)** is valued at approximately **$8.5M** on the open market. With this and salary data, I was able to to easily calculate
what players are over/underpaid relative to the WAR they bring their teams. 
* **Surplus Value Calculation:** `(WAR × $8.5M) − Contract AAV`
* **Strategic Goals:**
* 1. Show MLB fans and GM's which players are producing well relative to their contract and which aren't. 
  2. Identify up-and-coming young players with elite underlying traits whose financial cost is significantly lower than their on-field production.

---

## 3 Main Parts

### 1. Central Dashboard

* **Top Surplus Values:** Identifies the best "bang-for-your-buck" assets.
* **Top Efficiency Values:** Identifies the worst players for the money they are paid.
* **Team Rankings:** Ranks organizations by total roster surplus. Higher +$'s is better.

### 2. Team Specific Views
This is similar to the central dashboard, but this provides a team specific view of high/low paid players + their breakout stars. 
* Worth mentioning that I excluded players without any AAV and I was only looking at position players, so you shouldn't see a player who's a pitcher or pre-arbitration.
* Had I included them, someone like Cam Smith who has a league minimum salary will look amazing because he produced 1 WAR (~8.5m in value). 

### 3. Breakout Stars
Ultimately, I wanted to figure out which players had not 'popped' yet but exhibited characteristics of those that had popped already. 
Specifically I wanted to focus on attributes that I feel players can most control as opposed to just telling them to get faster or stronger, which is easier said than done. 
As a result, I built a ranking system that evaluates position players across highly predictive metrics, mostly having to do with decision making and fundamentals:
* **Meatball Swing %:** Aggression on pitches in the heart of the zone.
* **Chase %:** How often players are chasing pitches that are balls, which generally result in poor contact.
* **Fast Swing % and Attack Angle:** Precision and speed of the swing plane.
* **IZ Contact:** How often they're hitting balls in the zone i.e are they hitting strikes

---

## 🛠️ Tech Stack
* **Data Sources:** Statcast, Baseball-Reference, Spotrac
* **App Site:** Streamlit
* Coded w/ Claude Code

---
