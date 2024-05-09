items = ["button", "slider", "switch", "chooser", "input", "monitor", "plot"]

def prompts_from_item(item, prompt_list):
    if item not in items:
        return None
    else:
        # get the prompts for each item
        prompts_item = [x for x in prompt_list if item in x['task_id']]
        return prompts_item
    
def item_from_task(task):
    # count how many times each item appears in the task and return the most frequent one\
    task = task.split()
    counter = {}
    for item in items:
        counter[item] = task.count(item)
    return max(counter, key=counter.get)

    
def task_to_prompt(task, prompt_list):
    item = item_from_task(task)
    prompts_item = prompts_from_item(item, prompt_list)
    
    prompt = """ You are a helpful assistant and based on the examples below, convert tasks into prompts. You use coordinates numerical values based on the examples, and assume the appropriate numerical values of the variables based on what would make sense, so there is no replacing to do. Return the prompt only in the format from the examples. For the coordinates use values from the examples, not  x1 y1 x2 y2 or similar."""
    for prompt_item in prompts_item:
        prompt += f"Example prompt: {prompt_item['prompt']}/n"
    return prompt
    
def second_prompt(first_prompt, prompt_list):
    item = item_from_task(first_prompt)
    prompts_item = prompts_from_item(item, prompt_list)

    prompt = """ You are a helpful assistant and based on some examples of solutions to prompts below, create a solution for this new prompt. Stick to the format that is used in the examples.For the coordinates use values from the examples, not  x1 y1 x2 y2 or similar."""
    for prompt_item in prompts_item:
        prompt += f"Prompt: {prompt_item['prompt']}/n"
        prompt += f"Solution: {prompt_item['solution']}/n"
    return prompt
    
    
def add_item_to_nlogo_interface(item, template):
    """Takes an item and adds it to the nlogo file under the interface section

    Args:
        item (str): The item to add to the nlogo interface
        template (str): The nlogo file template
    """
    splitted_template = template.split('@#$#@#$#@\n')
    splitted_template[1] += f'\n{item}\n'
    
    # create str with the new added item
    # Export again to a .nlogo file
    new_nlogo = '\n@#$#@#$#@\n'.join(splitted_template)
    
    return  new_nlogo