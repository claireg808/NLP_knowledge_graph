   --- Initial Environment Setup (Do Once) ---
    Create a new virtual environment: 
        - python -m venv myenv 
        Replace "myenv" with your desired name
    Activate the environment to install necessary dependencies:
        - source myenv/bin/activate
    Install these dependencies:
        pip install --upgrade pip
        torch==2.5.1 
        llm==0.7.3



    --- Establish DS Connection & Generate JSONs ---
    Run the following script first to get DS running:
        - sbatch run.sh
    Check the job status in output_serve.log: 
        - tail -f output_serve.log
        Wait for this line to appear: 
            - INFO:     Application startup complete.
    Run 'generate_json.py': 
        - sbatch generate_json.sh
        A folder titled 'platinum_relations' will be populated
        Prints 'Complete' when done



    --- Connect To Neo4j From Hickory ---
    (Initial set-up) 
        Download Neo4j desktop app
            Local instances -> Create instance
            Create an instance name and password
            The default username should be 'neo4j'
        Install the APOC plug-in (Three dots -> Plugins -> Install APOC)
        Open the neo4j.conf file (Three dots -> Open neo4j.conf)
            Add, update, and/or uncomment these lines:
                dbms.security.auth_enabled=true
                dbms.security.procedures.unrestricted=apoc.meta.*,apoc.*
                dbms.security.procedures.allowlist=apoc.meta.*,apoc.*
                server.cluster.system_database_mode=PRIMARY
                server.default_listen_address=0.0.0.0
                server.bolt.enabled=true
                server.bolt.listen_address=0.0.0.0:7687
                server.bolt.advertised_address=:7687
                server.http.enabled=true
                server.http.listen_address=0.0.0.0:7474
                server.config.strict_validation.enabled=false
                apoc.import.file.enabled=true
                apoc.export.file.enabled=true
        Create a .env file with these lines:
            URI = bolt://localhost:9687
            USERNAME = neo4j
            PASSWORD = <your-password-here>
        NOTE: Add .env to .gitignore!!

    Start your local instance Neo4j database via the Desktop app

    Run this line on your machine's CLI (insert your username): 
        - ssh -R 9687:localhost:7687 <username>@hickory.cs.vcu.edu
    Test that the connection was successful:
        - nc -zv localhost 9687
        This line should be returned: 
            - Connection to localhost (127.0.0.1) 9687 port [tcp/*] succeeded!
    Keep this port open while using Neo4j (stay signed into Hickory/ keep the terminal open)

    On VSCode, in Hickory, run this to check the Neo4j connection (insert your password):
python -c "from neo4j import GraphDatabase
try:
    driver = GraphDatabase.driver('bolt://localhost:9687', auth=('neo4j', '<your-password-here>'))                           
    driver.verify_connectivity()
    print('Neo4j connection working on remote server!')
    driver.close()
except Exception as e:
    print(f'Connection failed: {e}')"

    Run 'generate_kg.py':
        - sbatch kg.sh



    --- General Notes ---
    To check the current hickory queue: squeue
    To cancel your job: scancel <job number>