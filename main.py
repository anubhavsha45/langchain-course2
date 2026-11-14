from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

def main():
    print("Hello from langchain-course 2")

    information="""
    Mahendra Singh Dhoni ([məˈɦeːnd̪ɾə ˈsɪŋɡʱ ˈd̪ʱoːniː] ⓘ; born 7 July 1981) is an Indian professional cricketer who plays as a right-handed batter and a wicket-keeper. Widely regarded as one of the most prolific wicket-keeper batsmen and captains, he represented the Indian cricket team and was the captain of the team in limited overs formats from 2007 to 2017 and in Test cricket from 2008 to 2014. Dhoni has captained the most number of international matches and is the most successful Indian captain. He has led India to victory in the 2007 ICC World Twenty20, the 2011 Cricket World Cup, and the 2013 ICC Champions Trophy, being the only captain to win three different limited overs ICC tournaments. He also led the teams that won the Asia Cup in 2010 and 2016, and was a member of the title-winning squad in 2018.

Born in Ranchi, Dhoni made his first-class debut for Bihar in 1999. He made his debut for the Indian cricket team on 23 December 2004 in an ODI against Bangladesh and played his first Test a year later against Sri Lanka. In 2007, he became the captain of the ODI team before taking over as captain in all formats by 2008. Dhoni retired from Test cricket in 2014 but continued playing in limited overs cricket until 2019. He has scored 17,266 runs in international cricket, including 10,773 runs at an average of more than 50 in ODIs.

In the Indian Premier League (IPL), Dhoni plays for the Chennai Super Kings (CSK), leading them to the finals on ten occasions and winning the title five times (2010, 2011, 2018, 2021, and 2023)– a record he jointly shares with Rohit Sharma. He has also led CSK to two Champions League T20 titles in 2010 and 2014. Dhoni is among the few batsmen to have scored more than five thousand runs in the IPL, as well as being the first wicket-keeper to do so.

In 2008, Dhoni was awarded India's highest sporting honour, the Khel Ratna Award, by the Government of India. He received the fourth highest civilian award, the Padma Shri, in 2009 and third highest civilian award, the Padma Bhushan, in 2018. Dhoni holds an honorary rank of Lieutenant Colonel in the Parachute Regiment of the Indian Territorial Army which was presented to him by the Indian Army in 2011. In June 2025, he was inducted into the ICC Cricket Hall of Fame.
    """

    summary_template="""
     given the information {information} about a person i want you to create
     1. A short summary
     2. Two interesting facts about them
"""

    summary_prompt_template=PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm=ChatGroq(
        temperature=0,
        model="openai/gpt-oss-120b"
    )

    chain=summary_prompt_template|llm

    response=chain.invoke({"information":information})

    print(response.content)



if __name__=="__main__":
    main()

