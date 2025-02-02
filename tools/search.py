from config import SERPAPI_API_KEY
import requests
from urllib.parse import urlencode

# -----------------------------
# Define Search Functions (Tools)
# -----------------------------
def search_google(query, start_date=None, end_date=None, num_results=50):
    if not query:
        return {'error': 'query is required'}

    base_url = 'https://serpapi.com/search.json'
    params = {
        'engine': 'google',
        'q': query,
        'api_key': SERPAPI_API_KEY
    }

    # If both start_date and end_date are provided, construct tbs parameter
    if start_date and end_date:
        # Example format: tbs=cdr:1,cd_min:01/01/2022,cd_max:12/31/2022
        tbs_value = f"cdr:1,cd_min:{start_date},cd_max:{end_date}"
        params['tbs'] = tbs_value

    search_url = f"{base_url}?{urlencode(params)}"

    try:
        response = requests.get(search_url)
        response.raise_for_status()
        search_results = response.json()

        if not search_results.get('organic_results'):
            return {'error': 'No results found'}

        # Process and limit results
        limited_results = search_results['organic_results'][:num_results]
        formatted_results = [{
            'title': result.get('title'),
            'link': result.get('link'),
            'snippet': result.get('snippet'),
            'date': result.get('date', 'Date unavailable').replace(', +0000 UTC', '')
        } for result in limited_results]

        return formatted_results

    except requests.exceptions.RequestException as e:
        return {'error': f'SerpAPI request error: {str(e)}'}
    except Exception as e:
        return {'error': f'Unexpected error: {str(e)}'}

SEARCH_GOOGLE_TOOL_METADATA = {
        "type": "function",
        "function": {
            "name": "search_google",
            "description": "Search Google for a given query, returning top results. Optionally specify a date range.",
            "parameters": {
            "type": "object",
            "properties": {
                "query": {
                "type": "string",
                "description": "The search query."
                },
                "start_date": {
                "type": "string",
                "description": "The start date for filtering search results (MM/DD/YYYY). If omitted, date filtering won't be applied."
                },
                "end_date": {
                "type": "string",
                "description": "The end date for filtering search results (MM/DD/YYYY). If omitted, date filtering won't be applied."
                }
            },
            "required": ["query"],
            "additionalProperties": False
            }
        }
    }