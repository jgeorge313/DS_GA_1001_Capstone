# Contributors Folder

In order to document each member's contribution to the overall project, as well as to minimize the risk of losing previously completed work, the team will upload individual ipython notebook files into the GitHub repository. 

When a team member has completed a corresponding section of the assignment, they will create a new branch and upload their most recent Jupyter Notebook to a branch in the following naming convention: **MemberName_vi (task completed).**

This way, for each branch, team members can quickly understand what each other has done, and are able to quickly back track to a previous branch in the case of errors. 


## Helpful tips on collaborative working

### Reading Data directly from github

This [stackoverflow](https://stackoverflow.com/questions/55240330/how-to-read-csv-file-from-github-using-pandas) describes how to read a csv file directly from GitHub. The following code can be used for our sample dataset:

    url = 'https://raw.githubusercontent.com/jgeorge313/DS_GA_1001_Capstone/main/Levels_Fyi_Salary_Data.csv'
    data = pd.read_csv(url)
    
### Running scripts in Jupyter notebook

To keep notebooks clean, we are creating helper notebooks that define functions used in our analytical script. We can execute these helper scripts in our analytical scripts, following this [stackoverflow](https://stackoverflow.com/questions/40054672/how-to-save-code-file-on-github-and-run-on-jupyter-notebook/48587645) we observe:

    !curl -Os <raw_github_script_url>
    %run <script_name.ipynb>

