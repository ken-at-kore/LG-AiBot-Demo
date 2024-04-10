import requests
import json
import re
import copy
from typing import List
from typing import Tuple
import traceback
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from datetime import datetime
import openai
import time



class AiBot:
    """
    AiBot applies the Facade design pattern to encapulate the OpenAI GPT calls.
    """

    def __init__(self, welcome_message:str, assistant_functions:List[dict]={}, run_step_callback=None):
        """
        Initialize the AiBot. Retreive the OpenAI GPT Assistant object and create an Assistant Thread.
        """

        self.welcome_message = welcome_message
        self.assistant_functions = assistant_functions
        self.run_step_callback = run_step_callback

        # Get the OpenAI Assistant key (which specifies the GPT model)
        assistant_preference = st.secrets.get("OPENAI_ASSISTANT_PREFERENCE", "OPENAI_GPT_3_5_ASSISTANT_ID")
        assert assistant_preference in st.secrets, f"OpenAI Assistant ID {assistant_preference} not found in secrets.toml file."
        openai_assistant_id = st.secrets[assistant_preference]

        # Initialize OpenAI Assistant objects
        self.assistant = openai.beta.assistants.retrieve(openai_assistant_id)
        print(f"AiBot: Assistant data retrieved.")
        self.bot_thread = openai.beta.threads.create()
        print(f"AiBot: Assistant Thread initialized.")

        # Initialize internal configs
        self.do_cot = False
        self.max_function_errors_on_turn = 1
        self.max_main_gpt_calls_on_turn = 4
        self.current_run_step_id = None



    def get_convo_messages(self):
        thread_messages = openai.beta.threads.messages.list(self.bot_thread.id).data
        return list(reversed(thread_messages))

    def add_user_message(self, user_message):
        openai.beta.threads.messages.create(self.bot_thread.id, role="user", content=user_message)


    
    def run(self) -> 'RunResult':
        
        # Initialize processing counters
        self.function_error_count = 0
        self.call_and_process_count = 0

        return self.call_and_process_gpt()

    def call_and_process_gpt(self, looping_run_id=None, function_outputs=None) -> 'RunResult':
        """
        Call the OpenAI GPT Assistant Run command to compute a chat completion then process the results.
        """
        # Keep track of how often call & processing happens this turn
        self.call_and_process_count += 1

        # Call OpenAI GPT (Response is streamed)
        gpt_call_count_this_round = 0
        bot_content_response = None
        required_func_calls = None
        while True:
            try:
                gpt_call_count_this_round += 1
                run_id, bot_content_response, required_func_calls = self.call_gpt(looping_run_id, function_outputs)
                break

            # Handle errors / timeouts
            except Exception as e:
                if gpt_call_count_this_round < 3:
                    print("AiBot: GPT call had an error. Trying again.")
                    traceback.print_exc()
                else:
                    print("AiBot: GPT calls had too many errors. Displaying error message.")
                    error_message = "Sorry, there was an error. Please try again."
                    bot_content_response = error_message
                    break
                
        there_are_required_func_calls = required_func_calls is not None and len(required_func_calls) > 0

        # Handle no function call
        if not there_are_required_func_calls:
            print(f"AiBot: GPT assistant message: {bot_content_response}")
            return AiBot.RunResult(bot_content_response)

        # Handle function call & display message
        else:
            # Add bot content to UI messages
            if bot_content_response != "":
                st.session_state.messages.append({"role": "assistant", "content": bot_content_response})

            # Handle the function call
            return self.handle_required_function_calls(required_func_calls, run_id)



    def call_gpt(self, looping_run_id=None, function_outputs=None) -> Tuple[str, str, List[dict], str]:
        """
        Call the OpenAI GPT Assistant Run command (either Create or Submit Tool Ouputs) and wait for completion.
        """
        bot_content_response = ""
        required_func_calls = []

        if function_outputs is None or function_outputs == []:
            print("AiBot: Creating Thread Run...")
            self.current_run_step_id = None
            run_creation = openai.beta.threads.runs.create(thread_id=self.bot_thread.id, assistant_id=self.assistant.id)

        else:
            print("AiBot: Submitting function output...")
            assert looping_run_id is not None, "Error: Expected a looping_run_id in call_gpt"
            run_creation = openai.beta.threads.runs.submit_tool_outputs(
                thread_id=self.bot_thread.id, run_id=looping_run_id, tool_outputs=function_outputs
            )
            print("AiBot: Output submission Run created.")

        print("AiBot: Polling status...")
        while True:
            time.sleep(0.2)
            run = openai.beta.threads.runs.retrieve(thread_id=self.bot_thread.id, run_id=run_creation.id)
            # print("AiBot: Run status: " + run.status)
            if run.status not in ['queued', 'in_progress']:
                break
            elif self.run_step_callback is not None:
                run_steps = openai.beta.threads.runs.steps.list(thread_id=self.bot_thread.id, run_id=run_creation.id).data
                self.process_run_steps_for_callback(run_steps)

        print("AiBot: Run done. Status: " + run.status)

        if run.status == 'completed':
            messages = openai.beta.threads.messages.list(self.bot_thread.id)
            print(f"AiBot: Retrieving new assistant message text. Messages count: {len(messages.data)}")
            bot_content_response = messages.data[0].content[0].text.value

        elif run.status == 'requires_action':
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            print("AiBot: Got required tool calls: " + str(tool_calls))
            for tool_call in tool_calls:
                assert tool_call.type == "function", "Error: Expected run tool_call to be a function call"
                required_func_calls.append({'id': tool_call.id, 
                                            'function_name': tool_call.function.name, 
                                            'function_args': tool_call.function.arguments}
                )
        elif run.status == 'failed':
            last_error_message = run.last_error.message
            print(f"AiBot: Run failed. {last_error_message} \n{run}")
            bot_content_response = f"Error: {last_error_message}"

        else:
            print("AiBot: Got an unexpected Run status. Run: " + str(run))
            raise Exception("Error: Got an unexpected Run terminal status: " + run.status)

        return (run.id, bot_content_response, required_func_calls)
    
    def process_run_steps_for_callback(self, run_steps):
        
        try:
            if len(run_steps) > 0:
                run_step = run_steps[0]
                if self.current_run_step_id is None or run_step.id != self.current_run_step_id:
                    print(f"AiBot: New run step: {run_step.type}")
                    self.current_run_step_id = run_step.id
                    tool_calls = None
                    if run_step.type == 'tool_calls':
                        tool_calls = []
                        for tool_call in run_step.step_details.tool_calls:
                            if tool_call['type'] == 'function':
                                tool_calls.append(('function', tool_call['function']['name']))
                            else:
                                tool_calls.append((tool_call['type'], None))
                    self.run_step_callback(run_step.type, tool_calls)
        except Exception as e:
            print(f"AiBot: Caught error when processing run steps: {e}")
            traceback.print_exc()

    

    def handle_required_function_calls(self, required_func_calls:List, run_id:str) -> 'RunResult':
        """
        Handle required function calls by executing the function and then re-running the GPT.
        """
        print("AiBot: Handling required function calls")

        # Execute and process all required function calls
        function_outputs = []
        there_was_a_func_call_error = False
        for required_func_call in required_func_calls:

            # Execute the function call
            func_call_result = self.execute_function_call(required_func_call['function_name'], required_func_call['function_args'])
            print(f"AiBot: Function execution result: {func_call_result.value}")

            # Check for errors
            if func_call_result.is_an_error_result:
                there_was_a_func_call_error = True

            # Append results to the outputs
            function_outputs.append({'tool_call_id': required_func_call['id'], 'output': func_call_result.value})

        if there_was_a_func_call_error:
            self.function_error_count += 1

        # Recursively call this same function to process the function call results
        return self.call_and_process_gpt(run_id, function_outputs)
    
    def execute_function_call(self, function_call_name:str, function_call_args:str) -> 'FuncResult':
        """
        Execute the function call. Catch exceptions.
        """        
        try:
            print("AiBot: Executing " + function_call_name)
            assert function_call_name in self.assistant_functions, f'Function {function_call_name} is not defined in the assistant function dictionary.'
            func_call_result = self.assistant_functions[function_call_name](json.loads(function_call_args))
            assert isinstance(func_call_result, AiBot.FuncResult), f"func_call_results for {function_call_name} must be of type AiFunction.Result, not {type(func_call_result)}"
        except Exception as e:
            error_info = f"{e.__class__.__name__}: {str(e)}"
            print(f"AiBot: Error executing function {function_call_name}: '{error_info}'")
            traceback.print_exc()
            func_call_result = AiBot.FuncResult(f"Caught exception when executing function {function_call_name}: '{error_info}'", is_an_error_result=True)

        return func_call_result
    


    class RunResult:
        def __init__(self, value:str):
            self.value = value



    class FuncResult:
        def __init__(self, value:str, is_an_error_result=False):
            self.value = value
            self.is_an_error_result = is_an_error_result









class StAiBot:
    """
    Streamlit AiBots orchestrate the various OpenAI GPT Assistant calls, execute function calls, and handle the StreamlitUI.
    """

    @staticmethod
    def initialize(streamlit_page_title:str, 
                   welcome_message:str,
                   assistant_functions:dict={},
                   function_display_descriptions:dict={},
                   embed_renderers:dict={}
        ):
        print("\n\n\n| - + - + - + - + - + - + - + - |\nAiBot: Initializing session.")
        bot = StAiBot(streamlit_page_title=streamlit_page_title, 
                        welcome_message=welcome_message, 
                        assistant_functions=assistant_functions,
                        function_display_descriptions=function_display_descriptions,
                        embed_renderers=embed_renderers
        )
        st.session_state['StAiBot'] = bot


    
    @staticmethod
    def is_initialized():
        return 'StAiBot' in st.session_state



    def __init__(self, streamlit_page_title:str, welcome_message:str, 
                 assistant_functions:dict={}, function_display_descriptions:dict={},
                 embed_renderers:dict={}):
        """
        Initialize the Streamlit AiBot. Construct the AiBot and initialize the UI.
        """

        # Initialize status UI variables, etc.
        self.status_ui = None
        self.function_display_descriptions = function_display_descriptions
        self.embed_renderers = embed_renderers

        # Initialize the AiBot
        self.aibot = AiBot(welcome_message, assistant_functions, self.run_step_callback)

        # Initialize the title
        self.streamlit_page_title = streamlit_page_title
        self.welcome_message = welcome_message

        # Initialize the message history with the welcome message
        self.message_history = []
        self.add_to_message_history({"text": welcome_message})

        # Streamlit wants to know the model though I don't think it uses it
        st.session_state["openai_model"] = self.aibot.assistant.model

        # Initialize the Streamlit page caption
        self.streamlit_page_caption = "Powered by Kore.ai."
        if 'gpt-3.5' in self.aibot.assistant.model:
            self.streamlit_page_caption += " (Model 3.5)"
        elif 'gpt-4' in self.aibot.assistant.model:
            self.streamlit_page_caption += " (Model 4)"



    @staticmethod
    def runBot():
        """
        Get the AiBot from the Streamlit session and run it.
        """
        bot = st.session_state['StAiBot']
        assert bot is not None, "StreamlitAiBot has not been initialized"
        bot.run()


    
    def run(self):
        """
        Run AiBot's main loop. The bot takes a turn.
        """
        print("AiBot: Running.")

        self.render_page()

        if user_input := st.chat_input("Enter text here"):

            self.add_to_message_history({"user-text": user_input})
            self.process_user_input(user_input)

            # TODO: Add the print_convo stuff

            with st.status("") as status:
                self.status_ui = status
                result = self.aibot.run()
                self.process_run_result(result)

            self.status_ui = None
            st.rerun()



    def add_to_message_history(self, message:dict):
        message['content_element_id'] = f"cont_elem_{len(self.message_history)}"
        self.message_history.append(message)
    
    
    
    def process_user_input(self, user_input):

        print(f"AiBot: User input: {user_input}")

        # Display user input
        with st.chat_message("user"):
            sanitized_user_input = user_input.replace('$','\$') # Sanitize against LaTeX markup
            st.markdown(sanitized_user_input) # Render it

        # Add the user input to the OpenAI Assistant Thread
        print("AiBot: Adding user message.")
        self.aibot.add_user_message(user_input)



    def run_step_callback(self, run_step_type, run_step_tool_calls):
        if self.status_ui is not None:
            label = ""
            if run_step_type == 'message_creation':
                label = "Typing..."
            elif run_step_type == 'tool_calls' and len(run_step_tool_calls) > 0:
                first_call = run_step_tool_calls[0]
                if first_call[0] == 'code_interpreter':
                    label = "Analyzing..."
                elif first_call[0] == 'retrieval':
                    label = "Reading docs..."
                elif first_call[0] == 'function':
                    func_name = first_call[1]
                    if func_name in self.function_display_descriptions:
                        label = f"{self.function_display_descriptions[func_name]}..."
                    else:
                        label = "Loading..."
                    
            self.status_ui.update(label=label)


    
    def process_run_result(self, run_result:AiBot.RunResult):
        """
        Process JSON run results. Sanatize JSON, extract content elements, and record to history.
        """
        json_signals = ["{", "}", "content-elements"]
        if all(s in run_result.value for s in json_signals):
            response_json = StAiBot.strip_non_json_parts(run_result.value)
            content_elements = json.loads(response_json)['content-elements'] #TODO: Catch parse errors here
            for element in content_elements:
                #TODO: Add validation checks
                self.add_to_message_history(element)
        else:
            self.add_to_message_history({"text": run_result.value})

    @staticmethod
    def strip_non_json_parts(text):
        """
        Check for the specified patterns in the text and strip out the unwanted parts.
        """
        # Remove ```json and ```.
        pattern = r"```json\s*\{.*?\}\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group().replace("```json", "").replace("```", "").strip()

        # Convert emoji to unicode representations
        def escape_non_ascii(match):
            return json.dumps(match.group(0))[1:-1]
        non_ascii_pattern = re.compile(r'[^\x00-\x7F]+') # Regex to match non-ASCII characters
        text = non_ascii_pattern.sub(escape_non_ascii, text)

        return text



    def render_page(self):

        # Display title and caption
        st.title(self.streamlit_page_title)
        if self.streamlit_page_caption is not None:
            st.caption(self.streamlit_page_caption)

        # New code
        for message in self.message_history:
            if "user-text" in message:
                with st.chat_message("user"):
                    st.markdown(message["user-text"].replace('$','\$')) # Sanitize against LaTeX markup
            elif "text" in message:
                with st.chat_message("assistant"):
                    st.markdown(message["text"].replace('$','\$')) # Sanitize against LaTeX markup
            elif "embed" in message and message["embed"] in self.embed_renderers: #TODO: Handle when renderer not found
                renderer = self.embed_renderers[message["embed"]]
                renderer(message["arguments"], message["content_element_id"])


        






# ------------------------------------------------------------------ #
# ------------------------ LG SPECIFIC CODE ------------------------ #

class LGProductInterface:

    def __init__(self):

        self.model_info_cache = {}
        self.renderer_cache = {}

        # Set up function calling callbacks
        def search_for_lg_products_callback(args):
            return self.search_for_lg_products(args)
        self.execute_search_for_lg_products = search_for_lg_products_callback
        def get_lg_product_details_callback(args):
            return self.get_lg_product_details(args)
        self.execute_get_lg_product_details = get_lg_product_details_callback

        # Set up embedded content element callbacks
        def render_product_gallery_callback(args, embed_id):
            self.render_product_gallery(args, embed_id)
        self.execute_render_product_gallery = render_product_gallery_callback
        def render_product_comparison_callback(args, embed_id):
            self.render_product_comparison(args, embed_id)
        self.execute_render_product_comparison = render_product_comparison_callback

    def search_for_lg_products(self, args) -> 'AiBot.FuncResult':
        """
        Search for LG products that match a semantic search and metadata filter criteria. Return summary info for a handful of results.
        """

        search_query = args['search_criteria']
        product_category = args['product_category']

        if 'price_filter' in args:
            price_filter = {"$and":[json.loads(args['price_filter']),{"price":{"$ne":0}}]}
        else:
            price_filter = {"price": {"$ne": 0}}

        embedding_vector = self._get_embedding(search_query)
        product_results = self._query_pinecone_db(embedding_vector, namespace=product_category, filter=price_filter)

        if 'matches' not in product_results:
            if 'message' in product_results:
                return AiBot.FuncResult(product_results['message'], is_an_error_result=True)
            else:
                return AiBot.FuncResult(str(product_results), is_an_error_result=True)

        results = []
        for vector_db_match in product_results['matches'][:6]:
            sku = vector_db_match['metadata']['sku']
            dynamo_product_info = self._get_product_data_from_dynamo_db(sku, reduce_to_summary_info=True)

            if not dynamo_product_info:
                print(f"LG AiBot: Warning: Couldn't find DynamoDB product info for {sku}")
                continue

            dynamo_product_info['product_url'] = 'https://www.lg.com/us' + dynamo_product_info['pdpUrl']
            del dynamo_product_info['pdpUrl']

            price_info = dynamo_product_info['price']
            dynamo_product_info['price'] = {
                'msrp': price_info['msrp']['displayText'],
                'final_price': price_info['finalPrice']['displayText'],
                'discount': price_info['discount']['displayText']
            }

            self.model_info_cache[sku] = copy.deepcopy(dynamo_product_info)

            del dynamo_product_info['imageUrl']

            results.append(dynamo_product_info)

        return AiBot.FuncResult(str(results))
    
    def _get_embedding(self, input):
        """Use a Kore.ai Azure service to generate an embedding vector of the search query."""

        # Endpoint URL
        url = "https://lgretail2.openai.azure.com/openai/deployments/AdaEmbedding002/embeddings?api-version=2023-03-15-preview"
        
        # Make the POST request
        payload = {"input": input}
        headers = {"Content-Type":"application/json","api-key":"216ce4fb4d4c475b89d2c2c13fa5bd25"}
        response = requests.post(url, json=payload, headers=headers)
        
        # Extract the embedding vector from the response (assuming the response contains 'vector' key)
        embedding_vector = response.json()['data'][0]['embedding']
        
        return embedding_vector

    def _query_pinecone_db(self, query_vector, namespace, filter:dict=None, top_k=7):
        """Function to query Pinecone DB."""

        # Endpoint URL
        url = "https://lg-pim-prod-d03a67d.svc.us-east-1-aws.pinecone.io/query"
        
        # Parameters
        payload = {
            "vector": [query_vector],
            "topK": top_k,
            "includeMetadata": True,
            "includeValues": False,
            "namespace": namespace
        }
        if filter is not None:
            payload['filter'] = filter

        # Make the POST request
        headers = {"Content-Type":"application/json","Api-Key":"e95eb932-1e9f-4240-bc08-3afea1970d9e"}
        response = requests.post(url, json=payload, headers=headers)
        
        # Return the response (can be parsed as needed)
        return response.json()



    def get_lg_product_details(self, args) -> 'AiBot.FuncResult':
        """Given a SKU, return detailed product information for an LG product."""

        sku = args['product_sku']

        detailed_info = self._get_product_data_from_dynamo_db(sku)

        return_info = {}

        main_info = detailed_info['product']
        return_info['title'] = main_info['title']
        return_info['sku'] = main_info['sku']
        return_info['category'] = main_info['categoryInfo'][0]['categoryName']
        return_info['product_url'] = 'https://www.lg.com/us' + main_info['pdpUrl']

        price_info = main_info['price']
        return_info['price'] = {
            'msrp': price_info['msrp']['displayText'],
            'final_price': price_info['finalPrice']['displayText'],
            'discount': price_info['discount']['displayText']
        }
        return_info['imageUrl'] = main_info['mediaAssets'][0]['thumbnailUrl']
        return_info['key_features'] = [item['feature'] for item in main_info['keyFeatures']]
        return_info['rating'] = f"{main_info['review']['points']} ({main_info['review']['reviewers']} reviewers)"
        return_info['model_code'] = main_info['modelCode']
        return_info['model_status'] = main_info['modelStatus']
        return_info['stock_status'] = main_info['stockStatus']['statusCode']

        return_info['additional_info'] = {
            spec['subtitle']: [
                item['term'] if item['description'] == "Yes" else f"{item['term']}: {item['description']}"
                for item in spec['tableData']
            ] for spec in detailed_info['allInfo']
        }

        self.model_info_cache[sku] = copy.deepcopy(return_info)

        return AiBot.FuncResult(str(return_info))


    def _get_product_data_from_dynamo_db(self, sku:str, reduce_to_summary_info:bool = False):
        """" 
        Get LG product data from the DynamoDB.
        """

        # Endpoint URL
        url = "https://retailassist-poc.kore.ai/pimWrapper/v1/getItems"
        if not reduce_to_summary_info:
            url += '?metaData="true"'

        # Parameters
        headers = {
            'Stage': 'dev',
            'x-secret-key': 'edc6f2b0-0b7a-11ec-82a8-0242ac130003',
            'Content-Type': 'application/json'
        }
        data = {
            "ids": [sku],
            "PIMEnv": "lg-pim-uat"
        }
        if not reduce_to_summary_info:
            data["metaData"] = "true"

        # Make the POST request and parse it
        response = requests.post(url, headers=headers, json=data)
        parsed_response = json.loads(response.text)
        if sku in parsed_response:
            product_info = parsed_response[sku]
        else:
            product_info = None

        return product_info



    def render_product_gallery(self, args:dict, content_element_id:str):

        embed_models = args["models"]

        if len(embed_models) == 1:
            columns = st.columns([0.1, 0.8, 0.1])
            self._render_model(embed_models[0], columns[1])
        elif len(embed_models) == 2:
            columns = st.columns([0.1, 0.4, 0.4, 0.1])
            self._render_model(embed_models[0], columns[1])
            self._render_model(embed_models[1], columns[2])
        elif len(embed_models) == 3:
            columns = st.columns(3)
            for i in range(3):
                self._render_model(embed_models[i], columns[i])
        elif len(embed_models) > 3:
            self._render_paginated_gallery(embed_models, content_element_id)

    def _render_paginated_gallery(self, embed_models:List[dict], content_element_id:str):

        if content_element_id in self.renderer_cache:
            current_page_index = self.renderer_cache[content_element_id]['current_page_index']
        else:  
            self.renderer_cache[content_element_id] = {'current_page_index': 0}
            current_page_index = 0

        products_per_page = 3
        first_product_in_view = current_page_index * products_per_page
        has_prev_button = first_product_in_view > 0
        has_next_button = first_product_in_view + products_per_page < len(embed_models)

        col1, col2, col3 = st.columns([0.8, 0.1, 0.1])
        cur_page_display = current_page_index + 1
        pages = -(-len(embed_models) // products_per_page) # Round up the division of product count by products per page
        col1.markdown(f"""
            <style>
                .container {{
                    display: grid; align-items: center; height: 40px; width: 100%; color: #8b8d99;
                }}
                .text-right {{
                    text-align: right; width: 100%;
                }}
            </style>
            <div class="container">
                <div class="text-right">
                    Page {cur_page_display} / {pages}
                </div>
            </div>
            """, unsafe_allow_html=True)
        if col2.button("Prev", key=f'{content_element_id}_prev', disabled=(not has_prev_button)):
            self.renderer_cache[content_element_id]['current_page_index'] -= 1
            st.rerun()
        if col3.button("Next", key=f'{content_element_id}_next', disabled=(not has_next_button)):
            self.renderer_cache[content_element_id]['current_page_index'] += 1
            st.rerun()
        
        product_columns = st.columns(products_per_page)
        products_in_view = embed_models[first_product_in_view:first_product_in_view+products_per_page]
        for i in range(len(products_in_view)):
            column = product_columns[i]
            model_args = products_in_view[i]
            self._render_model(model_args, column)

    def _render_model(self, model_args:dict, container:DeltaGenerator):

        model_info = self.model_info_cache[model_args["model"]]

        container.image(model_info['imageUrl'])

        small_font_span = "<span style='font-size: 14px; color: #5B5B5B;'>"
        model_text = f"""{small_font_span}{model_info['sku']}</span>
        <br><a href="{model_info['product_url']}" target="_blank"><b>{model_info['title'].strip()}</b></a>"""
        if "note" in model_args:
            model_text += '<br>' + model_args["note"]
        model_text += f"<br>{small_font_span}{model_info['price']['final_price']}</span>"
        model_text = model_text.replace('$','\$') # Sanitize against LaTeX markup
        container.markdown(model_text, unsafe_allow_html=True)


    
    def render_product_comparison(self, args:dict, content_element_id:str):

        embed_models = args["models"]
        comparison_rows = args["comparison-rows"]

        table_width = len(embed_models) + 1 # +1 because the first cell is empty
        header_cells = st.columns(table_width)
        for i in range(len(embed_models)):
            header_cell = header_cells[i+1]
            embed_model = embed_models[i]
            self._render_model({"model": embed_model}, header_cell)

        for comparison_row in comparison_rows:
            row_cells = st.columns(table_width)
            row_cells[0].markdown(f"**{comparison_row['row-label']}**")
            for i in range(table_width-1):
                cell = row_cells[i+1]
                comparison = comparison_row["comparisons"][i]
                cell.markdown(comparison)

        

class MockLGCart:
    def __init__(self):
        self.cart = []
        def add_item_to_cart(args:dict):
            self.cart.append(args)
            call_response = f"Item added successfully. Cart: {self.cart}"
            return AiBot.FuncResult(call_response)
        self.add_item_to_cart_function = add_item_to_cart
        def remove_item_from_cart(args:dict):
            sku_to_remove = args['product_sku']
            self.cart = [item for item in self.cart if item['product_sku'] != sku_to_remove]
            call_response = f"Item removed successfully. Cart: {self.cart}"
            return AiBot.FuncResult(call_response)
        self.remove_item_from_cart_function = remove_item_from_cart









# ------------------------------------------------------------------ #
# --------------------------- RUN AIBOT ---------------------------- #

# Initialize the bot (if not initialized)
if not StAiBot.is_initialized():

    # Setup Streamlit page configs
    page_title = 'LG Chatbot'
    st.set_page_config(
        page_title=page_title,
        page_icon="🤖",
    )

    # Initialize LG objects
    lg_product_interface = LGProductInterface()
    mock_cart = MockLGCart()

    # Initialize the AIBot
    StAiBot.initialize(streamlit_page_title=page_title,
                        welcome_message=open('prompts & content/welcome message.md').read(),
                        assistant_functions={
                            'search_for_lg_products': lg_product_interface.execute_search_for_lg_products,
                            'get_lg_product_details': lg_product_interface.execute_get_lg_product_details,
                            'add_item_to_cart': mock_cart.add_item_to_cart_function,
                            'remove_item_from_cart': mock_cart.remove_item_from_cart_function
                        },
                        function_display_descriptions={
                            'search_for_lg_products': "Searching",
                            'get_lg_product_details': "Getting details",
                            'add_item_to_cart': "Adding",
                            'remove_item_from_cart': "Removing"
                        },
                        embed_renderers={
                            'product-showcase': lg_product_interface.execute_render_product_gallery,
                            'product-comparison-table': lg_product_interface.execute_render_product_comparison
                        }
    )

# Run the AIBot
StAiBot.runBot()
