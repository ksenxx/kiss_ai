

class BaseSea:

    def base_prompt(t):
        t = super().base_prompt(t)
        return self.prompt(t)

    def prompt(t):
        return t

    def base_system_prompt(system_prompt):
        t = super().base_system_prompt(t)
        return self.system_prompt(t)
    
    def system_prompt(system_prompt):
        return system_prompt

    def base_tools(tls):
        tls = super().base_tools(tls)
        return self.tools(tls)

    def tools(tls):
        return tls

    def base_tool_call_hook(names, args):
        response = super().base_tool_call_hook(name, args)
        if response != "OK":
            return response
        else:
            return self.tool_call_hook(nam, args)
    
    def tool_call_hook(name, args):
        return "OK"

    def base_llm_call_hook(new_messages):
        new_messages = super().base_llm_call_hook(new_messages)
        return self.llm_call_hook(new_messages)

    def llm_call_hook(new_messages):
        return new_messages

    def base_settings(s):
        s = super().base_settings(s)
        return self.settings(s)

    def settings(s):
        return s

        