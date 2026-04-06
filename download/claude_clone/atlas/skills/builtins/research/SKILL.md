---
name: research
description: Web research skill for gathering information from the web, synthesizing findings, and producing research summaries.
version: 1.0.0
tags: [web, research, information, search]
dependencies: []
author: Hermes Team
---

# Research Skill

When this skill is active, follow these steps to conduct thorough web research on a given topic.

## Overview

This skill provides a systematic approach to researching topics using web search, URL fetching, and information synthesis. Use it when the user needs comprehensive information about a subject.

## Steps

1. **Clarify the research question**: Identify the core question or topic. Break complex topics into sub-questions. Determine the scope (breadth vs. depth).

2. **Plan the search strategy**: Identify 3-5 key search queries that cover different aspects of the topic. Include both broad and specific queries. Consider alternative terms and synonyms.

3. **Execute web searches**: Use the web_search tool for each planned query. Collect at least 5-10 results per query. Prioritize authoritative sources (official docs, research papers, reputable sites).

4. **Deep-dive into promising sources**: Use fetch_url to read the most relevant results in full. Extract key facts, data points, quotes, and conclusions. Note the source URL and publication date for citations.

5. **Synthesize findings**: Organize information by theme or sub-topic. Identify agreements and disagreements between sources. Highlight gaps in available information. Note any conflicting claims that need further investigation.

6. **Produce research summary**: Structure the output with:
   - Executive summary (2-3 sentences)
   - Key findings (bullet points with source citations)
   - Detailed analysis by sub-topic
   - Unresolved questions or areas needing more research
   - Source list with URLs

## Expected Outcome

A comprehensive, well-sourced research summary that the user can use to make decisions or understand a topic deeply.
